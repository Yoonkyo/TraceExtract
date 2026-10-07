"""
curate_captions_agibot.py
=========================

AgiBot-tuned variant of `curate_captions.py`. AgiBot episodes are already
task-level segments, so this script applies a *conservative soft-tuning*
of the motion/acceleration chunking — only major action transitions
trigger a split, and most episodes end up as 1-3 chunks. The per-episode
`description.txt` is also injected into the VLM prompt as
`task_description`.

For each AgiBot episode folder under the dataset root, the pipeline:

  1. Loads `acceleration.npz` and partitions the timeline into
     contiguous motion chunks. Compared to `curate_captions.py`:
       * peak prominence requirement is much higher (only major motion
         events count),
       * `min_chunk_len` / `max_chunk_len` are larger so a single
         AgiBot episode is preferred,
       * `max_chunk_len` is treated as a soft safety guardrail rather
         than an aggressive splitter,
       * if no reliable major peaks are found, the whole episode
         becomes one chunk instead of being uniformly partitioned.
  2. Reads the per-episode `description.txt` once and threads it
     through to the VLM caption call as `task_description`.
  3. For each chunk, builds a motion mask image, then sends
     (first_frame, last_frame, mask, [middle_frame], task_description,
     segment metadata) to the multimodal model and parses a strict JSON
     schema. The prompt explicitly tells the model to use the task
     description as a hint only and to rely on visual evidence.
  4. Builds adjacent-only sliding windows and runs a text-only merge
     call per window, exactly as in the base pipeline.
  5. Saves intermediate debug artefacts under `curated/` AND a lean
     training-oriented export `curated_training_texts.json` that
     additionally carries `task_description`, `chunk_index`,
     `num_chunks`, `start_frame`, `end_frame`, and
     `selected_frame_indices` per chunk.

Run `python curate_captions_agibot.py --help` for usage.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# Optional but very useful — used only if available.
try:
    from scipy.signal import savgol_filter, find_peaks  # type: ignore

    _HAS_SAVGOL = True
    _HAS_FIND_PEAKS = True
except Exception:  # pragma: no cover - scipy is optional at runtime
    _HAS_SAVGOL = False
    _HAS_FIND_PEAKS = False

from dotenv import load_dotenv

# OpenAI async client. Imported lazily-friendly: failing here is a real error.
from openai import AsyncOpenAI
from openai import APIError, APIConnectionError, RateLimitError, BadRequestError

# Gemini SDK is optional — only imported when --backend gemini is used.
try:
    from google import genai as google_genai  # type: ignore
    from google.genai import types as google_genai_types  # type: ignore

    _HAS_GEMINI = True
except Exception:  # pragma: no cover
    _HAS_GEMINI = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("curate_captions")


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    root: Path
    backend: str = "openai"  # "openai" or "gemini"
    # Captioning vs merge use separate models so the cheaper / smaller
    # variant can be used for the text-only merge stage.
    caption_model: str = "gpt-4o-mini"
    merge_model: str = "gpt-4o-mini"
    # Chunking — AgiBot conservative defaults. Episodes are already
    # task-level segments, so we want 1-3 chunks per episode and only
    # split at major motion transitions.
    smooth_window: int = 11
    min_chunk_len: int = 30
    max_chunk_len: int = 300
    accel_threshold_ratio: float = 0.35  # frac of (smoothed_max - smoothed_min)
    min_prominence_ratio: float = 0.40   # min peak prominence as frac of range
    max_chunks: int = 3                  # hard cap on chunks per video
    pad_frames: int = 1
    # Mask
    mask_radius: int = 12
    mask_dilate: int = 5
    # AgiBot: whether to actually send the motion mask to the VLM. The
    # mask is still BUILT (so debug/inspection artefacts under
    # `curated/masks/` are unchanged) — this flag only controls whether
    # it is included in the multimodal request. Default is False because
    # the mask tends to over-focus the model on low-level arm motion at
    # the expense of task-relevant object state changes (e.g. a
    # refrigerator door swinging open).
    use_motion_mask: bool = False
    # Conditional middle frame
    min_frames_for_middle: int = 30      # chunk must be >= this many frames
    min_disp_for_middle: float = 5.0     # dominant cluster displacement (px)
    # Ego-motion filter: any frame with >= this many keypoints is marked
    # noisy and excluded from chunking. `noisy_pad` expands each noisy
    # interval by this many frames on each side before building clean
    # segments (useful if the filter classifier has one-frame flicker).
    max_keypoints_per_frame: int = 300
    noisy_pad: int = 0
    # API concurrency
    max_concurrent_requests: int = 6
    request_timeout: float = 60.0
    max_retries: int = 4
    dry_run: bool = False
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def episode_id_from_dir(d: Path) -> str:
    """Return the episode id used in inner filenames, e.g. 'episode_11980'."""
    return d.name


def discover_videos(root: Path) -> list[Path]:
    """Find subfolders that look like a processed episode directory.

    Requires the unified format: each video folder must contain
    ``acceleration.npz`` and ``images.npy``.
    """
    out: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "acceleration.npz").exists() and (child / "images.npy").exists():
            out.append(child)
    return out


def _has_valid_json(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        json.loads(path.read_text())
    except Exception:
        return False
    return True


def episode_has_completed_outputs(video_dir: Path) -> bool:
    """Return True when final outputs already exist and are parseable."""
    required = ("curated_captions.json", "curated_training_texts.json")
    for output_dir in (video_dir, video_dir / "curated"):
        if all(_has_valid_json(output_dir / name) for name in required):
            return True
    return False


def read_task_description(video_dir: Path) -> str:
    """Return the contents of ``description.txt`` for an AgiBot episode.

    Whitespace-stripped. Returns ``""`` when the file is missing or
    cannot be read for any reason — callers must tolerate an empty
    string.
    """
    path = video_dir / "description.txt"
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except Exception as e:  # unreadable / encoding issues
        logger.warning("could not read %s: %s", path, e)
        return ""


# ---------------------------------------------------------------------------
# Step 1 — chunking by acceleration
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: int
    start_idx: int
    end_idx: int
    peak_acceleration: float
    mean_acceleration: float
    segment_id: int = 0  # index of the clean segment this chunk came from

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx + 1

    def to_meta(self, video_id: str) -> dict[str, Any]:
        return {
            "video_id": video_id,
            "chunk_id": self.chunk_id,
            "segment_id": self.segment_id,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "length": self.length,
            "peak_acceleration": float(self.peak_acceleration),
            "mean_acceleration": float(self.mean_acceleration),
        }


def _smooth(acc: np.ndarray, window: int) -> np.ndarray:
    """Smooth a 1-D acceleration trace.

    Uses Savitzky-Golay when scipy is available and the window is large
    enough; otherwise falls back to a centered moving average. Both
    suppress per-frame jitter so the threshold step finds *motion events*
    rather than individual noisy spikes.
    """
    n = len(acc)
    if n == 0:
        return acc
    w = max(3, min(window, n if n % 2 == 1 else n - 1))
    if w % 2 == 0:
        w += 1
    if _HAS_SAVGOL and w >= 5 and n >= w:
        try:
            return savgol_filter(acc, window_length=w, polyorder=2)
        except Exception:
            pass
    # moving average fallback
    pad = w // 2
    padded = np.pad(acc, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(padded, kernel, mode="valid")


def _make_chunk(acc: np.ndarray, s: int, e: int, cid: int) -> Chunk:
    seg = acc[s : e + 1]
    return Chunk(
        chunk_id=cid,
        start_idx=int(s),
        end_idx=int(e),
        peak_acceleration=float(seg.max()) if len(seg) else 0.0,
        mean_acceleration=float(seg.mean()) if len(seg) else 0.0,
    )


def _enforce_chunk_lengths(
    boundaries: list[int],
    n: int,
    cfg: PipelineConfig,
    acc: np.ndarray,
) -> list[Chunk]:
    """Turn an ascending list of split frame indices into Chunks covering [0,n-1].

    `boundaries` are *interior* split frames: each split `b` means a new
    chunk starts at frame `b`. The function:
      * always prepends 0 and appends n so chunks tile the full range,
      * merges chunks shorter than `cfg.min_chunk_len` into a neighbour,
      * splits chunks longer than `cfg.max_chunk_len` into roughly-equal
        contiguous pieces.
    The result is a contiguous, non-overlapping, full-coverage partition.
    """
    pts = sorted({0, n, *(b for b in boundaries if 0 < b < n)})
    # build raw [start, end_inclusive] segments
    segs: list[list[int]] = [[pts[i], pts[i + 1] - 1] for i in range(len(pts) - 1)]

    # Merge too-short segments into a neighbour. We always merge into the
    # smaller-acceleration neighbour to keep big motion chunks intact.
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (s, e) in enumerate(segs):
            if e - s + 1 >= cfg.min_chunk_len:
                continue
            # Pick the neighbour to absorb us into.
            left_ok = i > 0
            right_ok = i < len(segs) - 1
            if left_ok and right_ok:
                # absorb into the shorter neighbour so we don't blow max_chunk_len
                left_len = segs[i - 1][1] - segs[i - 1][0] + 1
                right_len = segs[i + 1][1] - segs[i + 1][0] + 1
                target = i - 1 if left_len <= right_len else i + 1
            elif left_ok:
                target = i - 1
            elif right_ok:
                target = i + 1
            else:
                break
            if target == i - 1:
                segs[i - 1][1] = e
            else:
                segs[i + 1][0] = s
            segs.pop(i)
            changed = True
            break

    # Split too-long segments into ~equal pieces.
    out_segs: list[tuple[int, int]] = []
    for s, e in segs:
        length = e - s + 1
        if length <= cfg.max_chunk_len:
            out_segs.append((s, e))
            continue
        n_splits = (length + cfg.max_chunk_len - 1) // cfg.max_chunk_len
        base = length // n_splits
        rem = length % n_splits
        cur = s
        for k in range(n_splits):
            sub_len = base + (1 if k < rem else 0)
            ss = cur
            ee = cur + sub_len - 1
            out_segs.append((ss, ee))
            cur = ee + 1

    chunks = [_make_chunk(acc, s, e, cid) for cid, (s, e) in enumerate(out_segs)]
    return chunks


def _find_peaks_with_prominence(
    smoothed: np.ndarray,
    cfg: PipelineConfig,
) -> list[int]:
    """Return peak indices sorted by prominence (descending).

    Uses scipy.signal.find_peaks when available; otherwise a manual
    fallback that computes prominence as the height above the higher of
    the two nearest valleys.
    """
    n = len(smoothed)
    s_min, s_max = float(smoothed.min()), float(smoothed.max())
    span = s_max - s_min
    if span < 1e-9:
        return []

    threshold = s_min + cfg.accel_threshold_ratio * span
    min_prom = cfg.min_prominence_ratio * span

    if _HAS_FIND_PEAKS:
        idxs, props = find_peaks(
            smoothed,
            height=threshold,
            distance=cfg.min_chunk_len,
            prominence=min_prom,
        )
        if len(idxs) == 0:
            return []
        # sort by prominence descending
        order = np.argsort(-props["prominences"])
        return [int(idxs[i]) for i in order]

    # Manual fallback: simple local-max peak finder with prominence check.
    raw_peaks: list[tuple[float, int]] = []  # (prominence, index)
    for i in range(1, n - 1):
        if smoothed[i] < threshold:
            continue
        if smoothed[i] < smoothed[i - 1] or smoothed[i] < smoothed[i + 1]:
            continue
        # compute prominence: scan left and right to find valleys
        left_min = smoothed[i]
        for j in range(i - 1, -1, -1):
            left_min = min(left_min, smoothed[j])
            if smoothed[j] > smoothed[i]:
                break
        right_min = smoothed[i]
        for j in range(i + 1, n):
            right_min = min(right_min, smoothed[j])
            if smoothed[j] > smoothed[i]:
                break
        prom = smoothed[i] - max(left_min, right_min)
        if prom >= min_prom:
            raw_peaks.append((float(prom), i))

    # sort by prominence descending
    raw_peaks.sort(key=lambda t: -t[0])

    # enforce minimum distance: greedily keep peaks in prominence order
    kept: list[int] = []
    for _, idx in raw_peaks:
        if all(abs(idx - k) >= cfg.min_chunk_len for k in kept):
            kept.append(idx)
    return kept


def chunk_acceleration(
    acc: np.ndarray,
    cfg: PipelineConfig,
    segment_start: int = 0,
    segment_end: int | None = None,
) -> list[Chunk]:
    """Partition a clean acceleration segment into contiguous motion chunks.

    Operates on the slice ``acc[segment_start : segment_end + 1]``; all
    returned chunk indices are remapped to GLOBAL frame indices so
    downstream code never has to know a segment was sliced out.

    Goal: every frame in the segment belongs to exactly one chunk, while
    boundaries are still chosen by acceleration content (so chunks tend
    to align with the start/end of motion events). High-level steps:

      1. Smooth the (segment) acceleration trace.
      2. Find prominence-ranked peaks; keep at most ``max_chunks - 1``.
      3. Split at the valley between each adjacent pair of peaks.
      4. If no peaks are found, fall back to a uniform partition.
      5. Apply min/max length safeguards, then remap back to global
         indices.
    """
    if segment_end is None:
        segment_end = len(acc) - 1
    if segment_end < segment_start:
        return []
    local = np.asarray(acc[segment_start : segment_end + 1])
    n = int(len(local))
    if n == 0:
        return []
    if n <= max(cfg.min_chunk_len, 2):
        c = _make_chunk(local, 0, n - 1, 0)
        c.start_idx += segment_start
        c.end_idx += segment_start
        return [c]

    smoothed = _smooth(local.astype(np.float64), cfg.smooth_window)

    # Step 2: prominence-based peak finder. With AgiBot's higher
    # `min_prominence_ratio`, this returns ONLY truly major motion
    # events (small jitters are filtered out by the prominence check
    # inside `_find_peaks_with_prominence`).
    peaks = _find_peaks_with_prominence(smoothed, cfg)

    # Cap at max_chunks - 1 peaks → max_chunks chunks per segment. The
    # safety-net allowance from `n // max_chunk_len` is intentionally
    # NOT applied here so that `max_chunk_len` stays a soft guardrail
    # — it only kicks in via `_enforce_chunk_lengths` below for
    # pathologically long segments, never as an aggressive driver of
    # extra chunks.
    max_peaks = max(1, cfg.max_chunks - 1)
    if len(peaks) > max_peaks:
        peaks = peaks[:max_peaks]

    peaks.sort()

    boundaries: list[int] = []
    if not peaks:
        # AgiBot soft-tuning: if no reliable major motion peak exists,
        # treat the whole segment as one chunk (no artificial splits).
        # `_enforce_chunk_lengths` may still split it if it exceeds
        # `max_chunk_len` (the soft guardrail), but for a typical
        # AgiBot episode of ~80-120 frames vs. max_chunk_len=300 that
        # never fires.
        logger.info(
            "No major acceleration peaks (prominence_ratio=%.2f); "
            "keeping the whole segment as one chunk.",
            cfg.min_prominence_ratio,
        )
    else:
        # Place each boundary at the lowest-acceleration valley between
        # adjacent prominent peaks — i.e. the quietest moment between
        # two major action phases. This is exactly the existing
        # behavior; the conservative peak set just means there are
        # fewer of them.
        for a, b in zip(peaks, peaks[1:]):
            valley = a + int(np.argmin(smoothed[a : b + 1]))
            if valley > a and valley < b:
                boundaries.append(valley)

    chunks = _enforce_chunk_lengths(boundaries, n, cfg, local)
    # Remap local → global frame indices.
    for k, c in enumerate(chunks):
        c.chunk_id = k
        c.start_idx += segment_start
        c.end_idx += segment_start
    logger.debug(
        "chunking segment [%d,%d]: %d peaks → %d boundaries → %d chunks",
        segment_start, segment_end, len(peaks), len(boundaries), len(chunks),
    )
    return chunks


# ---------------------------------------------------------------------------
# Frame-level ego-motion filter  → clean segments
# ---------------------------------------------------------------------------


def find_noisy_intervals(
    per_frame_counts: np.ndarray,
    threshold: int,
    pad: int = 0,
) -> list[tuple[int, int]]:
    """Return sorted, merged ``[start, end]`` inclusive noisy intervals.

    A frame is noisy when ``per_frame_counts[t] >= threshold``.
    Contiguous noisy frames collapse into one interval; optional ``pad``
    frames are added on each side and overlapping intervals are merged.
    """
    n = int(len(per_frame_counts))
    if n == 0 or threshold <= 0:
        return []
    noisy = per_frame_counts >= threshold
    if not noisy.any():
        return []
    intervals: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not noisy[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and noisy[j + 1]:
            j += 1
        s = max(0, i - pad)
        e = min(n - 1, j + pad)
        if intervals and s <= intervals[-1][1] + 1:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], e))
        else:
            intervals.append((s, e))
        i = j + 1
    return intervals


def clean_segments_from_noisy(
    n_frames: int,
    noisy_intervals: list[tuple[int, int]],
    min_len: int,
) -> list[tuple[int, int]]:
    """Complement of ``noisy_intervals`` over ``[0, n_frames-1]``.

    Returns inclusive ``[start, end]`` segments whose length is at least
    ``min_len``. Segments shorter than ``min_len`` are dropped as
    unusable.
    """
    if n_frames <= 0:
        return []
    segments: list[tuple[int, int]] = []
    cursor = 0
    for s, e in noisy_intervals:
        if cursor <= s - 1:
            segments.append((cursor, s - 1))
        cursor = max(cursor, e + 1)
    if cursor <= n_frames - 1:
        segments.append((cursor, n_frames - 1))
    return [(a, b) for (a, b) in segments if (b - a + 1) >= min_len]


# ---------------------------------------------------------------------------
# Per-video data handles (unified .npy format)
# ---------------------------------------------------------------------------


class VideoData:
    """Mmap-backed handles for one video directory (unified .npy format).

    Assumes the directory contains:
      * ``images.npy``                    (T, H, W, 3) uint8
      * ``samples/frame_indices.npy``     (F,) int
      * ``samples/offsets.npy``           (F+1,) int
      * ``samples/keypoints.npy``         (N_total, 2) — moving keypoints only
      * ``samples/cluster_ids.npy``       (N_total,)
      * ``samples/raw_traj.npy``          (N_total, future_len, 3)
      * ``samples/raw_valid_steps.npy``   (N_total, future_len)

    Exposes:
      * ``get_rgb(t)``        → ``(H, W, 3)`` uint8 ndarray
      * ``get_sample(t)``     → dict of per-keypoint arrays (or ``None``)
      * ``num_keypoints_at(t)`` → number of keypoints indexed at frame *t*
      * ``save_rgb_to(t, path)`` → write frame *t* as a PNG
      * ``n_frames``          → total number of RGB frames

    Heavy arrays are opened with ``mmap_mode="r"`` so the OS only pages in
    the rows actually sliced. Handles are opened once per video and
    reused across all chunks.
    """

    def __init__(self, video_dir: Path, video_id: str):
        self.video_dir = video_dir
        self.video_id = video_id

        self._images = np.load(video_dir / "images.npy", mmap_mode="r")
        self.n_frames = int(self._images.shape[0])
        self._image_h = int(self._images.shape[1])
        self._image_w = int(self._images.shape[2])

        samples_dir = video_dir / "samples"
        self._frame_indices = np.load(samples_dir / "frame_indices.npy")
        self._offsets = np.load(samples_dir / "offsets.npy")
        self._kp = np.load(samples_dir / "keypoints.npy", mmap_mode="r")
        self._cids = np.load(samples_dir / "cluster_ids.npy", mmap_mode="r")
        self._raw_traj = np.load(samples_dir / "raw_traj.npy", mmap_mode="r")
        self._raw_vs = np.load(samples_dir / "raw_valid_steps.npy", mmap_mode="r")
        # Lookup: frame_index → slot in frame_indices
        self._frame_to_slot: dict[int, int] = {
            int(t): i for i, t in enumerate(self._frame_indices)
        }

    # --- RGB access ----------------------------------------------------------

    def get_rgb(self, t: int) -> np.ndarray:
        """Return frame *t* as ``(H, W, 3)`` uint8."""
        return np.array(self._images[t])

    def has_frame(self, t: int) -> bool:
        return 0 <= t < self.n_frames

    def save_rgb_to(self, t: int, path: Path) -> None:
        """Write frame *t* as a PNG at *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.get_rgb(t)).save(path)

    # --- sample / keypoint access --------------------------------------------

    def num_keypoints_at(self, t: int) -> int:
        """Return the number of keypoints indexed at frame *t*, or 0."""
        slot = self._frame_to_slot.get(t)
        if slot is None:
            return 0
        return int(self._offsets[slot + 1]) - int(self._offsets[slot])

    def per_frame_keypoint_counts(self) -> np.ndarray:
        """Return an int array of length ``n_frames`` with per-frame counts.

        Frames that have no indexed sample (absent from
        ``frame_indices.npy``) get count 0.
        """
        counts = np.zeros(self.n_frames, dtype=np.int64)
        diffs = np.diff(self._offsets).astype(np.int64)
        fi = np.asarray(self._frame_indices, dtype=np.int64)
        valid = (fi >= 0) & (fi < self.n_frames)
        counts[fi[valid]] = diffs[valid]
        return counts

    def get_sample(self, t: int) -> dict[str, np.ndarray] | None:
        """Return keypoint data for frame *t*, or ``None`` if unavailable."""
        slot = self._frame_to_slot.get(t)
        if slot is None:
            return None
        lo = int(self._offsets[slot])
        hi = int(self._offsets[slot + 1])
        if hi <= lo:
            return None
        return {
            "keypoints": np.array(self._kp[lo:hi], dtype=np.float32),
            "cluster_ids": np.array(self._cids[lo:hi]),
            "raw_traj": np.array(self._raw_traj[lo:hi], dtype=np.float32),
            "raw_valid_steps": np.array(self._raw_vs[lo:hi]),
        }


# ---------------------------------------------------------------------------
# Step 3a — motion mask construction
# ---------------------------------------------------------------------------

# We always load the sample at the *chunk start*: raw_traj already covers
# the full chunk span (start..end), and `keypoints` gives the chunk-start
# pixel positions, which is what we want to rasterize the mask on. The
# dominant moving cluster is the one with the largest
# ``mean_2d_displacement * sqrt(cluster_size)`` from raw_traj[:,0] to
# raw_traj[:, end-start].
#
# ``raw_traj`` has a fixed ``future_len`` dimension padded with -inf;
# ``raw_valid_steps`` masks out the padding. Preprocessing now stores only
# moving keypoints in ``keypoints.npy`` / ``cluster_ids.npy``, so there is
# no per-point ``is_moving`` flag to filter on.


@dataclass
class MaskResult:
    mask_path: Path
    overlay_path: Path | None
    cluster_id: int | None
    n_keypoints: int
    displacement: float = 0.0  # mean 2-D px displacement of dominant cluster
    notes: str = ""


def _pick_dominant_cluster(
    sample: dict[str, np.ndarray],
    chunk_span: int,
) -> tuple[int | None, float]:
    """Return ``(cluster_id, mean_displacement)`` for the dominant cluster.

    ``sample`` is the keypoint data at the chunk *start* frame.
    ``chunk_span`` is ``end_idx - start_idx`` — the offset into
    ``raw_traj`` for the chunk end position.

    Scoring is ``mean_2d_displacement * sqrt(cluster_size)`` so that
    spatially extended moving regions are preferred over a few
    high-velocity outlier points.
    """
    raw_traj = sample.get("raw_traj")
    cluster_ids = sample.get("cluster_ids")
    if raw_traj is None or cluster_ids is None:
        return None, 0.0
    raw_traj = np.asarray(raw_traj, dtype=np.float32)  # (K, T, 3)
    cluster_ids = np.asarray(cluster_ids).astype(int)  # (K,)
    if raw_traj.ndim != 3 or raw_traj.shape[0] != cluster_ids.shape[0]:
        return None, 0.0
    K, T, _ = raw_traj.shape
    s = 0
    e = int(np.clip(chunk_span, 1, T - 1))
    if e <= s:
        return None, 0.0

    valid = sample.get("raw_valid_steps")
    if valid is not None:
        valid = np.asarray(valid).astype(bool)
        ok = valid[:, s] & valid[:, e]
    else:
        ok = np.ones(K, dtype=bool)

    # Mask out -inf entries in raw_traj (padding).
    finite_s = np.isfinite(raw_traj[:, s, :2]).all(axis=-1)
    finite_e = np.isfinite(raw_traj[:, e, :2]).all(axis=-1)
    ok &= finite_s & finite_e

    if not ok.any():
        return None, 0.0

    disp = np.linalg.norm(raw_traj[:, e, :2] - raw_traj[:, s, :2], axis=-1)
    best_cid: int | None = None
    best_score = -1.0
    best_disp = 0.0
    for cid in np.unique(cluster_ids):
        m = ok & (cluster_ids == cid)
        n = int(m.sum())
        if n < 4:  # need a real region, not a couple of outliers
            continue
        mean_d = float(disp[m].mean())
        score = mean_d * (n ** 0.5)
        if score > best_score:
            best_score = score
            best_cid = int(cid)
            best_disp = mean_d
    return best_cid, best_disp


def _draw_mask(
    keypoints: np.ndarray,
    image_size: tuple[int, int],
    radius: int,
    dilate: int,
) -> Image.Image:
    """Rasterize a binary mask by drawing filled disks at *keypoints*."""
    w, h = image_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for x, y in keypoints:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cx, cy = float(x), float(y)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    if dilate > 0:
        # cheap dilation: paste mask onto larger ellipses by re-drawing.
        # PIL has no native dilate; ImageFilter.MaxFilter is enough here.
        from PIL import ImageFilter

        # MaxFilter size must be odd >= 3
        size = max(3, dilate * 2 + 1)
        if size % 2 == 0:
            size += 1
        mask = mask.filter(ImageFilter.MaxFilter(size))
    return mask


def _save_overlay(rgb: Image.Image, mask: Image.Image, path: Path) -> None:
    rgb_rgba = rgb.convert("RGBA")
    color = Image.new("RGBA", rgb_rgba.size, (255, 64, 64, 0))
    alpha = mask.point(lambda v: 120 if v > 0 else 0)
    color.putalpha(alpha)
    overlay = Image.alpha_composite(rgb_rgba, color)
    overlay.convert("RGB").save(path)


def build_motion_mask(
    vdata: VideoData,
    chunk: Chunk,
    out_dir: Path,
    cfg: PipelineConfig,
) -> MaskResult:
    """Build the motion mask + overlay PNGs for a chunk.

    Uses the ``VideoData`` handle so the caller controls mmap lifetime.

    Failure modes are handled gracefully:
      * sample missing → empty mask, ``cluster_id=None``
      * malformed sample → same, plus a note in MaskResult.notes
      * dominant cluster has no points → fall back to all keypoints
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the chunk-start frame: raw_traj at frame i covers exactly the
    # forward window [i .. N-1], so the start sample contains the full chunk
    # span and its keypoints already match the start RGB frame.
    start_idx = chunk.start_idx
    if not vdata.has_frame(start_idx):
        for cand in range(chunk.start_idx, chunk.end_idx + 1):
            if vdata.has_frame(cand):
                start_idx = cand
                break
    if not vdata.has_frame(start_idx):
        raise FileNotFoundError(f"No RGB frame found for chunk {chunk.chunk_id}")

    rgb_arr = vdata.get_rgb(start_idx)
    rgb = Image.fromarray(rgb_arr)
    image_size = rgb.size  # (w, h)

    sample = vdata.get_sample(start_idx)
    notes = ""
    cluster_id: int | None = None
    chunk_span = chunk.end_idx - start_idx  # offset into raw_traj for chunk end

    displacement = 0.0

    if sample is None:
        notes = f"missing or unreadable sample at frame {start_idx}"
        keypoints = np.zeros((0, 2), dtype=np.float32)
    else:
        try:
            cluster_id, displacement = _pick_dominant_cluster(sample, chunk_span)
            kp_all = np.asarray(sample["keypoints"], dtype=np.float32)
            cids = np.asarray(sample["cluster_ids"]).astype(int)
            if cluster_id is None:
                notes = "no dominant cluster; using all keypoints"
                m = np.ones(len(cids), dtype=bool)
            else:
                m = cids == cluster_id
                if m.sum() < 2:
                    notes = f"cluster {cluster_id} too small; falling back to all keypoints"
                    m = np.ones(len(cids), dtype=bool)
            keypoints = kp_all[m]
        except Exception as e:
            notes = f"sample parse error: {e}"
            keypoints = np.zeros((0, 2), dtype=np.float32)

    mask = _draw_mask(keypoints, image_size, cfg.mask_radius, cfg.mask_dilate)
    mask_path = out_dir / f"chunk_{chunk.chunk_id:03d}_mask.png"
    overlay_path = out_dir / f"chunk_{chunk.chunk_id:03d}_overlay.png"
    mask.save(mask_path)
    try:
        _save_overlay(rgb, mask, overlay_path)
    except Exception as e:
        logger.warning("overlay save failed for chunk %d: %s", chunk.chunk_id, e)
        overlay_path = None  # type: ignore

    return MaskResult(
        mask_path=mask_path,
        overlay_path=overlay_path,
        cluster_id=cluster_id,
        n_keypoints=int(len(keypoints)),
        displacement=displacement,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Step 2/3 — multimodal captioning
# ---------------------------------------------------------------------------


CHUNK_SYSTEM_PROMPT = """You are a careful vision annotator for a
cross-embodiment 3D-trajectory-prediction dataset. You receive RGB
images for one short motion clip, optionally a motion-mask image, and
(for AgiBot episodes) an overall `task_description` plus segment
metadata as additional text context:

  1. FIRST_FRAME — the RGB frame at the start of the clip.
  2. LAST_FRAME  — the RGB frame at the end of the clip.
  3. (Optional) MIDDLE_FRAME — a frame near the temporal midpoint of
     the clip, included only for longer or higher-displacement chunks.
  4. (Optional) MOTION_MASK — a binary mask (white pixels) highlighting
     the dominant moving region in this clip. The mask, when present,
     is an auxiliary attention hint only; it does not override the RGB
     evidence or the task description.

No meta-language (critical):
  * NEVER use meta-language that refers to the video, image inputs, or
    annotation artifacts. Forbidden phrases include: "in the first
    frame", "in the middle frame", "in the last frame", "the masked
    region", "the motion mask", "FIRST_FRAME", "LAST_FRAME",
    "MIDDLE_FRAME". Describe the physical objects and motions directly
    as they happen in the real world.

Use of `task_description`:
  * The episode-level `task_description` gives the high-level goal and
    the target object/action. Use it to disambiguate what the robot is
    likely trying to do — but the caption must describe the SPECIFIC
    visible phase in this segment, not the whole task.
  * Different segments of the same episode MUST receive different
    captions. If two adjacent segments would describe the same visible
    change, the wording is too generic — re-anchor on the concrete
    pixel-level change between this segment's start and end.
  * Do NOT copy or near-paraphrase the task description verbatim. Pick
    the correct target object / action from the task description only
    when it is visually plausible in the segment.
  * Avoid overly low-level kinematic wording (e.g. "the arm moves
    forward") when a task-relevant object or state change is visible.
  * Use `hallucination_flags` when the target object/action is not
    visually supported or the segment is genuinely ambiguous, not
    merely because a segment shows a partial step toward the goal.

Three instruction styles (each MUST play a different role):
  * `instruction_1` — SEGMENT-SPECIFIC visual description. Describe
    only what visibly changes between the start and end of THIS
    segment (third-person, present tense). It must NOT be a generic
    paraphrase of the episode-level task. Two adjacent segments
    should yield two different `instruction_1` strings.
  * `instruction_2` — Task-oriented imperative robot command for THIS
    segment, using the episode goal as context. Imperative style,
    ideally <= 30 words. Refer to the target object by the noun the
    task description uses when it is visually plausible.
  * `instruction_3` — Natural human-like request, as if casually
    spoken to a robot. May sit closer to the overall task goal
    (since users phrase requests at goal-level), but it MUST NOT
    contradict what is visible in this segment. Ideally <= 20 words.

Embodiment classification:
  * Classify the acting entity in the `embodiment_type` field.
    Choose exactly one of: "human_hand", "robot_gripper", "robot_arm",
    "tool", "unknown".
  * In natural-language text (`instruction_1`, `instruction_2`,
    `instruction_3`), you MAY use embodiment-specific wording when
    visually supported — e.g. "the human hand", "the robot gripper",
    "the robotic arm". Keep wording CONSISTENT with the selected
    `embodiment_type`.
  * When the embodiment is unclear, set `embodiment_type` to "unknown"
    and use the generic phrase "the manipulator" in text.
  * If the moving region is primarily on a manipulated OBJECT rather
    than on the manipulator itself, describe the object's motion using
    task-aware wording and do not force an embodiment label at all.

Skill labels (closed taxonomy):
  * Choose 1 to 2 labels from this exact set:
    ["reach", "grasp", "lift", "transport", "place", "release",
     "push", "pull", "rotate", "open_close", "reposition", "unknown"]
  * Labels correspond to manipulation phases:
    - reach: the manipulator approaches an object (closing distance,
      no contact yet).
    - grasp: the manipulator closes around or makes contact with an
      object (object and manipulator overlap, object begins to move
      with manipulator).
    - lift / transport: an object is held and moves through space
      (both move together, object is above its resting surface).
    - place / release: the manipulator lowers or releases an object
      to a target location (object descends, manipulator opens or
      retracts).
    - push / pull / rotate / open_close / reposition: other
      single-step skills.
  * If the skill is ambiguous, use ["unknown"] — do NOT force a label.

Hallucination awareness:
  * You MUST set the `hallucination_flags` fields honestly:
    - `contact_claim_without_clear_evidence`: true if you describe
      grasping, contact, or picking up but you are not certain from
      the pixels alone.
    - `object_identity_uncertain`: true if you named or described an
      object but its identity is not visually clear.
    - `embodiment_uncertain`: true if you picked an embodiment_type
      but the evidence is not strong.
  * When ANY hallucination flag is true, make the natural-language
    text MORE cautious and LESS assertive — use hedging language
    ("appears to", "seems to", "possibly").

Hard rules:
  * Focus on the action visible in the RGB frames. If a motion mask is
    provided, treat it as an auxiliary attention hint — never let it
    override the task description or visible object state changes in
    the RGB frames.
  * Compare the start and end images pixel by pixel — describe motion
    and object state changes that are visible. When a middle image is
    provided, use it to disambiguate the trajectory (e.g. whether
    the motion is a straight path or an arc).
  * The task description is the episode-level goal; use it to
    disambiguate object identities and intent when visually plausible.
    Do not, however, invent details (specific brand, color, count)
    that are absent from both the task description and the visible
    scene.
  * Do not speculate about causes outside the frame.
  * If something is genuinely ambiguous (visual evidence inconsistent
    with the task description, or scene unreadable), say so explicitly
    and set "confidence" accordingly.
  * Output strictly valid JSON, no markdown, no commentary.
"""


CHUNK_USER_PROMPT_3IMG = """You are given 3 images: FIRST_FRAME, LAST_FRAME, and MOTION_MASK.

Return JSON with exactly this schema:

{
  "instruction_1": "<segment-specific visual description: describe ONLY what visibly changes between this segment's start and end (third-person). Two adjacent segments must yield two different instruction_1 strings. Do not paraphrase the full task description.>",
  "instruction_2": "<task-oriented imperative robot command for THIS segment, using the episode goal as context, ideally <= 30 words>",
  "instruction_3": "<natural human-like request spoken casually to a robot; may sit closer to the overall task goal but must not contradict the visible segment, ideally <= 20 words>",
  "embodiment_type": "<human_hand|robot_gripper|robot_arm|tool|unknown>",
  "task_type": "manipulation",
  "skill_labels": ["<1-2 labels from: reach, grasp, lift, transport, place, release, push, pull, rotate, open_close, reposition, unknown>"],
  "hallucination_flags": {
    "contact_claim_without_clear_evidence": <true|false>,
    "object_identity_uncertain": <true|false>,
    "embodiment_uncertain": <true|false>
  },
  "structured_motion": {
    "main_actor": "<short noun phrase; use natural embodiment wording consistent with embodiment_type, or 'the manipulator' if unknown>",
    "moving_region": "<where in the scene the moving area is, e.g. 'upper-right table area'>",
    "start_state": "<what the moving region looks like / where it is at the start>",
    "end_state": "<same, at the end>",
    "motion_type": "<reach|grasp|lift|transport|place|release|translate|rotate|push|pull|lower|tilt|idle|none|unclear>",
    "motion_direction": "<left|right|up|down|forward|backward|diagonal|rotational|none|unclear>",
    "temporal_pattern": "<smooth|accelerating|decelerating|brief|sustained|stop-start|unclear>",
    "interaction": "<what touches what, only if clearly visible; otherwise 'unclear'>",
    "confidence": "<low|medium|high>"
  }
}

Remember: NEVER use meta-language referring to frames, images, or masks.
Describe the physical scene directly. Output JSON only.
"""

CHUNK_USER_PROMPT_4IMG = """You are given 4 images: FIRST_FRAME, MIDDLE_FRAME, LAST_FRAME, and MOTION_MASK.
The middle image is taken near the temporal midpoint of the clip. Use it
to understand the trajectory of the motion (e.g. whether movement is
linear, curved, or involves a direction change).

Return JSON with exactly this schema:

{
  "instruction_1": "<segment-specific visual description: describe ONLY what visibly changes between this segment's start and end (third-person). Two adjacent segments must yield two different instruction_1 strings. Do not paraphrase the full task description.>",
  "instruction_2": "<task-oriented imperative robot command for THIS segment, using the episode goal as context, ideally <= 30 words>",
  "instruction_3": "<natural human-like request spoken casually to a robot; may sit closer to the overall task goal but must not contradict the visible segment, ideally <= 20 words>",
  "embodiment_type": "<human_hand|robot_gripper|robot_arm|tool|unknown>",
  "task_type": "manipulation",
  "skill_labels": ["<1-2 labels from: reach, grasp, lift, transport, place, release, push, pull, rotate, open_close, reposition, unknown>"],
  "hallucination_flags": {
    "contact_claim_without_clear_evidence": <true|false>,
    "object_identity_uncertain": <true|false>,
    "embodiment_uncertain": <true|false>
  },
  "structured_motion": {
    "main_actor": "<short noun phrase; use natural embodiment wording consistent with embodiment_type, or 'the manipulator' if unknown>",
    "moving_region": "<where in the scene the moving area is, e.g. 'upper-right table area'>",
    "start_state": "<what the moving region looks like / where it is at the start>",
    "mid_state": "<what the moving region looks like / where it is at the midpoint>",
    "end_state": "<same, at the end>",
    "motion_type": "<reach|grasp|lift|transport|place|release|translate|rotate|push|pull|lower|tilt|idle|none|unclear>",
    "motion_direction": "<left|right|up|down|forward|backward|diagonal|rotational|none|unclear>",
    "temporal_pattern": "<smooth|accelerating|decelerating|brief|sustained|stop-start|unclear>",
    "interaction": "<what touches what, only if clearly visible; otherwise 'unclear'>",
    "confidence": "<low|medium|high>"
  }
}

Remember: NEVER use meta-language referring to frames, images, or masks.
Describe the physical scene directly. Output JSON only.
"""


# ---- AgiBot mask-disabled variants -----------------------------------
# Used when `cfg.use_motion_mask` is False (the default for AgiBot).
# These are identical to the *_3IMG / *_4IMG templates above except that
# they no longer reference a MOTION_MASK image.

CHUNK_USER_PROMPT_2IMG_NOMASK = """You are given 2 images: FIRST_FRAME and LAST_FRAME.

Return JSON with exactly this schema:

{
  "instruction_1": "<segment-specific visual description: describe ONLY what visibly changes between this segment's start and end (third-person). Two adjacent segments must yield two different instruction_1 strings. Do not paraphrase the full task description.>",
  "instruction_2": "<task-oriented imperative robot command for THIS segment, using the episode goal as context, ideally <= 30 words>",
  "instruction_3": "<natural human-like request spoken casually to a robot; may sit closer to the overall task goal but must not contradict the visible segment, ideally <= 20 words>",
  "embodiment_type": "<human_hand|robot_gripper|robot_arm|tool|unknown>",
  "task_type": "manipulation",
  "skill_labels": ["<1-2 labels from: reach, grasp, lift, transport, place, release, push, pull, rotate, open_close, reposition, unknown>"],
  "hallucination_flags": {
    "contact_claim_without_clear_evidence": <true|false>,
    "object_identity_uncertain": <true|false>,
    "embodiment_uncertain": <true|false>
  },
  "structured_motion": {
    "main_actor": "<short noun phrase; use natural embodiment wording consistent with embodiment_type, or 'the manipulator' if unknown>",
    "moving_region": "<where in the scene the moving area is, e.g. 'upper-right table area'>",
    "start_state": "<what the moving region looks like / where it is at the start>",
    "end_state": "<same, at the end>",
    "motion_type": "<reach|grasp|lift|transport|place|release|translate|rotate|push|pull|lower|tilt|idle|none|unclear>",
    "motion_direction": "<left|right|up|down|forward|backward|diagonal|rotational|none|unclear>",
    "temporal_pattern": "<smooth|accelerating|decelerating|brief|sustained|stop-start|unclear>",
    "interaction": "<what touches what, only if clearly visible; otherwise 'unclear'>",
    "confidence": "<low|medium|high>"
  }
}

Remember: NEVER use meta-language referring to frames or images.
Describe the physical scene directly. Output JSON only.
"""

CHUNK_USER_PROMPT_3IMG_NOMASK = """You are given 3 images: FIRST_FRAME, MIDDLE_FRAME, and LAST_FRAME.
The middle image is taken near the temporal midpoint of the clip. Use it
to understand the trajectory of the motion (e.g. whether movement is
linear, curved, or involves a direction change).

Return JSON with exactly this schema:

{
  "instruction_1": "<segment-specific visual description: describe ONLY what visibly changes between this segment's start and end (third-person). Two adjacent segments must yield two different instruction_1 strings. Do not paraphrase the full task description.>",
  "instruction_2": "<task-oriented imperative robot command for THIS segment, using the episode goal as context, ideally <= 30 words>",
  "instruction_3": "<natural human-like request spoken casually to a robot; may sit closer to the overall task goal but must not contradict the visible segment, ideally <= 20 words>",
  "embodiment_type": "<human_hand|robot_gripper|robot_arm|tool|unknown>",
  "task_type": "manipulation",
  "skill_labels": ["<1-2 labels from: reach, grasp, lift, transport, place, release, push, pull, rotate, open_close, reposition, unknown>"],
  "hallucination_flags": {
    "contact_claim_without_clear_evidence": <true|false>,
    "object_identity_uncertain": <true|false>,
    "embodiment_uncertain": <true|false>
  },
  "structured_motion": {
    "main_actor": "<short noun phrase; use natural embodiment wording consistent with embodiment_type, or 'the manipulator' if unknown>",
    "moving_region": "<where in the scene the moving area is, e.g. 'upper-right table area'>",
    "start_state": "<what the moving region looks like / where it is at the start>",
    "mid_state": "<what the moving region looks like / where it is at the midpoint>",
    "end_state": "<same, at the end>",
    "motion_type": "<reach|grasp|lift|transport|place|release|translate|rotate|push|pull|lower|tilt|idle|none|unclear>",
    "motion_direction": "<left|right|up|down|forward|backward|diagonal|rotational|none|unclear>",
    "temporal_pattern": "<smooth|accelerating|decelerating|brief|sustained|stop-start|unclear>",
    "interaction": "<what touches what, only if clearly visible; otherwise 'unclear'>",
    "confidence": "<low|medium|high>"
  }
}

Remember: NEVER use meta-language referring to frames or images.
Describe the physical scene directly. Output JSON only.
"""


MERGE_SYSTEM_PROMPT = """You merge several chunk-level motion captions for
a contiguous slice of a cross-embodiment manipulation video into one
coherent description. You receive a JSON list of per-chunk captions in
temporal order. You MUST follow every rule below — violations are bugs.

No meta-language (critical):
  * NEVER use meta-language that refers to the video, image inputs, or
    annotation artifacts. Forbidden phrases include: "in the first
    frame", "in the middle frame", "in the last frame", "the masked
    region", "the motion mask", "chunk N". Describe the physical
    objects and motions directly as they happen in the real world.

Output text styles:
  * `instruction_1`: Concise action description covering the full
    merged sequence (descriptive third-person style). Ideally
    <= 35 words.
  * `instruction_2`: Step-by-step robot manipulation instruction for
    the full merged sequence (imperative command style). Ideally
    <= 30 words.
  * `instruction_3`: Natural human-like request for the full merged
    sequence, as if casually spoken to a robot. Ideally <= 20 words.

Faithfulness:
  * Do NOT introduce any object, action, intent, location, owner,
    interaction, cause, or effect that does not appear in the input
    chunk captions. The merge stage is text-only and you cannot see
    pixels — you have nothing to invent from.
  * If the input does not say something, do not say it.
  * If different chunks disagree, prefer hedged wording ("then it
    moves further", "the position changes again") over picking one.

Order:
  * Preserve strict chronological order. Never reorder chunks.
  * The timeline array must list chunks in the same order they appear
    in the input.
  * Each timeline entry MUST include `chunk_id`, `start_idx`, and
    `end_idx` copied verbatim from the input — do not invent or shift
    these numbers.

Embodiment wording:
  * Preserve the embodiment wording used in the input chunk captions.
    If the chunks consistently say "the robot gripper", keep that
    phrasing in the merged text.
  * If different chunks use different embodiment terms, unify to the
    most common term from the inputs or fall back to "the manipulator".

Conciseness:
  * Combine adjacent chunks that describe the same continuing motion
    into one timeline phrase rather than repeating yourself.
  * Remove redundancy without inventing new facts.
  * For long sequences, summarize the main trajectory and overall state
    change. Do not list every single micro-action or direction change.
    Avoid verbose chains such as "moves right, then moves left, then
    shifts again." Focus on the dominant object, main action, and final
    state. Keep the merged instructions punchy and training-friendly.

You must respond with a valid JSON object only. Do not include any other text.
Output strict JSON only — no markdown, no commentary.
"""


MERGE_USER_PROMPT_TEMPLATE = """Here are the per-chunk captions for one
contiguous time window, ordered by time:

{chunks_json}

Return JSON with this exact schema:

{{
  "instruction_1": "<concise action description covering the full sequence, descriptive third-person style, ideally <= 35 words>",
  "instruction_2": "<step-by-step robot manipulation instruction for the full sequence, imperative command style, ideally <= 30 words>",
  "instruction_3": "<natural human-like request for the full sequence, casual tone, ideally <= 20 words>",
  "skill_labels": ["<union of skill_labels across the merged chunks, deduplicated>"],
  "timeline": [
    {{
      "chunk_id": <int, copied verbatim from input>,
      "start_idx": <int, copied verbatim from input>,
      "end_idx": <int, copied verbatim from input>,
      "event": "<short phrase describing what happens in this chunk, ideally <= 12-15 words>"
    }}
  ]
}}

Do not add chunks that were not in the input. Do not drop chunks that
were in the input. NEVER use meta-language referring to frames, images,
masks, or chunk numbers. Describe the physical scene directly.
You must respond with a valid JSON object only. Do not include any other text.
Output JSON only.
"""


def build_task_context_block(
    task_description: str,
    chunk_index: int,
    num_chunks: int,
    start_frame: int,
    end_frame: int,
    *,
    use_motion_mask: bool = False,
) -> str:
    """Render the AgiBot task / segment context that prepends the user prompt.

    `chunk_index` is 0-based here; we display it as 1-based in the
    rendered text (matching how a human would read "segment 1 of 3").
    Returns an empty string when `task_description` is empty so the
    base prompt is sent unchanged.

    When `use_motion_mask=True`, an extra sentence is appended that
    reminds the model the mask is an auxiliary attention hint only,
    so it cannot override the task description or visible object
    state changes in the RGB frames.
    """
    if not task_description:
        return ""
    block = (
        "Overall task description:\n"
        f"\"{task_description}\"\n\n"
        "Segment information:\n"
        f"- Segment index: {chunk_index + 1} of {num_chunks}\n"
        f"- Frames covered: {start_frame} to {end_frame}\n\n"
        "Instruction:\n"
        "The task description gives the episode-level goal and target "
        "object/action. Use it to disambiguate what the robot is likely "
        "trying to do.\n\n"
        "However, the caption must describe the specific visible phase "
        "in this segment, not simply repeat or paraphrase the full "
        "task description.\n\n"
        "Guidelines:\n"
        "- Do not output the same generic task-level sentence for "
        "every segment.\n"
        "- Describe the concrete visual change between the start and "
        "end of this segment.\n"
        "- If the segment shows preparation, describe it as "
        "preparation toward the goal.\n"
        "- If the segment shows contact or object motion, describe the "
        "specific interaction and the resulting object state change.\n"
        "- If the segment shows completion or stabilization, describe "
        "the final state or release.\n"
        "- Use the task description to choose the correct target "
        "object/action when visually plausible, but avoid copying the "
        "task description verbatim.\n"
        "- Avoid overly low-level kinematic wording such as \"the arm "
        "moves forward\" unless no task-relevant object or state "
        "change is visible.\n"
        "- Use uncertainty flags when the target object/action is not "
        "visually supported or the segment is genuinely ambiguous.\n"
    )
    if use_motion_mask:
        block += (
            "\nA motion mask is provided only as an auxiliary attention "
            "hint. Do not let it override the overall task description "
            "or visible object state changes in the RGB frames.\n"
        )
    return block + "\n"


def encode_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _data_url(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".") or "png"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{encode_image_b64(path)}"


def _extract_json(text: str | None) -> dict[str, Any]:
    """Best-effort JSON extraction from a model response."""
    if text is None or not text.strip():
        logger.warning("LLM returned empty string; cannot parse JSON")
        raise json.JSONDecodeError("LLM returned empty string", text or "", 0)

    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    else:
        # Some models return stray backticks without a complete fenced block.
        # Fall back to the full raw string after trimming those wrappers.
        text = raw.strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # find the first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


# ---------------------------------------------------------------------------
# OpenAI client wrapper
# ---------------------------------------------------------------------------


class CaptionClient:
    """Thin wrapper around AsyncOpenAI with retries + JSON parsing.

    The interface is intentionally minimal so a Gemini-backed implementation
    can be dropped in by reimplementing `caption_chunk` and `merge_chunks`.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.client = AsyncOpenAI(timeout=cfg.request_timeout)
        self.semaphore = asyncio.Semaphore(cfg.max_concurrent_requests)

    @staticmethod
    def _supports_only_default_temperature(model: str) -> bool:
        lower = model.lower()
        return lower.startswith(("gpt-5", "o1", "o3", "o4"))

    @staticmethod
    def _uses_completion_token_limit(model: str) -> bool:
        lower = model.lower()
        return lower.startswith(("gpt-5", "o1", "o3", "o4"))

    def _chat_kwargs(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, str],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": response_format,
        }
        if self._uses_completion_token_limit(model):
            # Reasoning-family models may spend part of this budget before
            # emitting visible JSON; keep enough room to avoid empty content.
            kwargs["max_completion_tokens"] = max(max_tokens, 2048)
        else:
            kwargs["max_tokens"] = max_tokens
        if not self._supports_only_default_temperature(model):
            kwargs["temperature"] = temperature
        return kwargs

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    value = part.get("text") or part.get("content")
                else:
                    value = getattr(part, "text", None) or getattr(part, "content", None)
                if isinstance(value, str):
                    parts.append(value)
            return "".join(parts)
        return ""

    async def _chat_with_retry_with_reason(
        self, **kwargs: Any
    ) -> tuple[str, str | None]:
        """Like `_chat_with_retry` but also returns the OpenAI `finish_reason`.

        `finish_reason == "length"` means the model hit `max_tokens` mid-output
        — surfacing it lets callers (e.g. merge_chunks) trigger a fallback
        rather than parsing truncated JSON.
        """
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                async with self.semaphore:
                    resp = await self.client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                text = self._message_text(choice.message)
                finish_reason = getattr(choice, "finish_reason", None)
                if not text.strip():
                    logger.warning(
                        "OpenAI returned empty message content "
                        "(model=%s, finish_reason=%s)",
                        kwargs.get("model"),
                        finish_reason,
                    )
                return text, finish_reason
            except (RateLimitError, APIConnectionError, APIError) as e:
                last_exc = e
                # exponential backoff with jitter
                delay = min(20.0, (2 ** attempt) * 0.5 + random.random() * 0.5)
                logger.warning(
                    "API call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    self.cfg.max_retries,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
            except BadRequestError as e:
                # never useful to retry a 400 — surface it
                raise
        assert last_exc is not None
        raise last_exc

    async def _chat_with_retry(self, **kwargs: Any) -> str:
        text, _ = await self._chat_with_retry_with_reason(**kwargs)
        return text

    async def caption_chunk(
        self,
        first_frame: Path,
        last_frame: Path,
        mask_image: Path | None,
        middle_frame: Path | None = None,
        *,
        task_description: str = "",
        chunk_index: int = 0,
        num_chunks: int = 1,
        start_frame: int = 0,
        end_frame: int = 0,
        use_motion_mask: bool = True,
    ) -> dict[str, Any]:
        """Send the multimodal triplet (or quartet) and parse the JSON response.

        AgiBot extras: `task_description` + segment metadata are
        prepended to the user prompt as a context block when present.
        When `use_motion_mask` is False (or `mask_image` is None), the
        mask is not sent and the prompt is selected from the no-mask
        variants so the prompt text matches what the model actually
        receives.
        """
        send_mask = use_motion_mask and mask_image is not None
        if send_mask:
            base_prompt = CHUNK_USER_PROMPT_4IMG if middle_frame else CHUNK_USER_PROMPT_3IMG
        else:
            base_prompt = (
                CHUNK_USER_PROMPT_3IMG_NOMASK
                if middle_frame
                else CHUNK_USER_PROMPT_2IMG_NOMASK
            )
        context_block = build_task_context_block(
            task_description, chunk_index, num_chunks, start_frame, end_frame,
            use_motion_mask=send_mask,
        )
        user_prompt = context_block + base_prompt
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
            {"type": "text", "text": "FIRST_FRAME:"},
            {"type": "image_url", "image_url": {"url": _data_url(first_frame)}},
        ]
        if middle_frame is not None:
            content += [
                {"type": "text", "text": "MIDDLE_FRAME:"},
                {"type": "image_url", "image_url": {"url": _data_url(middle_frame)}},
            ]
        content += [
            {"type": "text", "text": "LAST_FRAME:"},
            {"type": "image_url", "image_url": {"url": _data_url(last_frame)}},
        ]
        if send_mask:
            content += [
                {"type": "text", "text": "MOTION_MASK:"},
                {"type": "image_url", "image_url": {"url": _data_url(mask_image)}},
            ]
        messages = [
            {"role": "system", "content": CHUNK_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        # one extra retry purely for malformed-JSON responses
        for json_attempt in range(2):
            text = await self._chat_with_retry(
                **self._chat_kwargs(
                    model=self.cfg.caption_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=700,
                )
            )
            try:
                return _extract_json(text)
            except json.JSONDecodeError as e:
                logger.warning("malformed JSON on attempt %d: %s", json_attempt + 1, e)
                if json_attempt == 1:
                    return {
                        "instruction_1": "",
                        "instruction_2": "",
                        "instruction_3": "",
                        "embodiment_type": "unknown",
                        "task_type": "manipulation",
                        "skill_labels": ["unknown"],
                        "hallucination_flags": {
                            "contact_claim_without_clear_evidence": False,
                            "object_identity_uncertain": False,
                            "embodiment_uncertain": True,
                        },
                        "structured_motion": {"confidence": "low"},
                        "_error": f"malformed JSON: {e}",
                        "_raw": text,
                    }
        # unreachable
        return {}

    # Model used as a robust fallback for merges. gpt-5-nano frequently
    # emits empty content, hits `finish_reason=length`, or returns
    # truncated JSON on longer windows; gpt-4o-mini handles them more
    # reliably and is also used directly for `merge_type == "full"`.
    MERGE_FALLBACK_MODEL = "gpt-4o-mini"

    async def _try_merge_call(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
    ) -> tuple[dict[str, Any] | None, str | None, str]:
        """Run one merge attempt. Returns (parsed_json, finish_reason, raw_text).

        `parsed_json` is None when the response is empty, hit the token
        limit (`finish_reason == "length"`), or could not be parsed as
        JSON. Callers can use that signal to fall back to another model.
        """
        text, finish_reason = await self._chat_with_retry_with_reason(
            **self._chat_kwargs(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=max_tokens,
            )
        )
        if not text.strip() or finish_reason == "length":
            return None, finish_reason, text
        try:
            return _extract_json(text), finish_reason, text
        except json.JSONDecodeError as e:
            logger.warning(
                "merge JSON malformed (model=%s, finish_reason=%s): %s",
                model, finish_reason, e,
            )
            return None, finish_reason, text

    async def merge_chunks(
        self,
        chunk_outputs: list[dict[str, Any]],
        *,
        merge_type: str = "",
    ) -> dict[str, Any]:
        prompt = MERGE_USER_PROMPT_TEMPLATE.format(
            chunks_json=json.dumps(chunk_outputs, indent=2)
        )
        messages = [
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Full-window merges go straight to gpt-4o-mini because they are
        # the longest / most failure-prone. Shorter merges use the
        # configured cheap model (e.g. gpt-5-nano).
        if merge_type == "full":
            primary_model = self.MERGE_FALLBACK_MODEL
            primary_max_tokens = 4096
        else:
            primary_model = self.cfg.merge_model
            primary_max_tokens = 2048

        parsed, finish_reason, raw = await self._try_merge_call(
            messages=messages,
            model=primary_model,
            max_tokens=primary_max_tokens,
        )
        if parsed is not None:
            logger.info(
                "merge ok via %s (merge_type=%s, n_chunks=%d)",
                primary_model, merge_type or "?", len(chunk_outputs),
            )
            return parsed

        # Fallback path: only if the primary wasn't already gpt-4o-mini.
        if primary_model == self.MERGE_FALLBACK_MODEL:
            logger.warning(
                "merge with %s failed (finish_reason=%s, merge_type=%s, "
                "n_chunks=%d); no further fallback available",
                primary_model, finish_reason, merge_type or "?",
                len(chunk_outputs),
            )
            return {
                "instruction_1": "",
                "instruction_2": "",
                "instruction_3": "",
                "timeline": [],
                "_error": (
                    f"merge failed with {primary_model} "
                    f"(finish_reason={finish_reason})"
                ),
                "_raw": raw,
            }

        logger.warning(
            "merge with %s failed (finish_reason=%s, merge_type=%s, "
            "n_chunks=%d); falling back to %s",
            primary_model, finish_reason, merge_type or "?",
            len(chunk_outputs), self.MERGE_FALLBACK_MODEL,
        )
        parsed_fb, finish_reason_fb, raw_fb = await self._try_merge_call(
            messages=messages,
            model=self.MERGE_FALLBACK_MODEL,
            max_tokens=4096,
        )
        if parsed_fb is not None:
            logger.info(
                "merge ok via fallback %s (merge_type=%s, n_chunks=%d)",
                self.MERGE_FALLBACK_MODEL, merge_type or "?",
                len(chunk_outputs),
            )
            return parsed_fb

        logger.error(
            "merge fallback %s also failed (finish_reason=%s, "
            "merge_type=%s, n_chunks=%d)",
            self.MERGE_FALLBACK_MODEL, finish_reason_fb,
            merge_type or "?", len(chunk_outputs),
        )
        return {
            "instruction_1": "",
            "instruction_2": "",
            "instruction_3": "",
            "timeline": [],
            "_error": (
                f"both primary ({primary_model}) and fallback "
                f"({self.MERGE_FALLBACK_MODEL}) failed; "
                f"primary_finish={finish_reason}, "
                f"fallback_finish={finish_reason_fb}"
            ),
            "_raw": raw_fb,
        }


# ---------------------------------------------------------------------------
# Gemini backend (drop-in alternative to CaptionClient)
# ---------------------------------------------------------------------------


class GeminiCaptionClient:
    """Gemini-backed implementation of the same interface as CaptionClient.

    Public methods (`caption_chunk`, `merge_chunks`) match `CaptionClient`
    exactly so the per-video pipeline does not need to know which backend is
    in use. Concurrency, retries, and JSON parsing mirror the OpenAI version.
    """

    def __init__(self, cfg: PipelineConfig):
        if not _HAS_GEMINI:
            raise RuntimeError(
                "google-genai is not installed; cannot use --backend gemini"
            )
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment")
        self.cfg = cfg
        self.client = google_genai.Client(api_key=api_key)
        self.semaphore = asyncio.Semaphore(cfg.max_concurrent_requests)

    async def _generate_with_retry(
        self,
        contents: list[Any],
        system_instruction: str,
        max_tokens: int,
        model: str,
    ) -> str:
        last_exc: Exception | None = None
        # Gemini 2.5 Flash spends "thinking" tokens before any visible output
        # and they count against maxOutputTokens — that silently truncates
        # JSON. We disable thinking for these structured-output calls and
        # also give the visible response a generous ceiling.
        config = google_genai_types.GenerateContentConfig(
            systemInstruction=system_instruction,
            temperature=0.2,
            maxOutputTokens=max(max_tokens, 1024),
            responseMimeType="application/json",
            thinkingConfig=google_genai_types.ThinkingConfig(thinkingBudget=0),
        )
        for attempt in range(self.cfg.max_retries):
            try:
                async with self.semaphore:
                    resp = await asyncio.wait_for(
                        self.client.aio.models.generate_content(
                            model=model,
                            contents=contents,
                            config=config,
                        ),
                        timeout=self.cfg.request_timeout,
                    )
                # `resp.text` joins all text parts of the first candidate.
                return resp.text or ""
            except asyncio.TimeoutError as e:
                last_exc = e
                logger.warning(
                    "Gemini call timed out (attempt %d/%d)",
                    attempt + 1,
                    self.cfg.max_retries,
                )
            except Exception as e:
                last_exc = e
                # Treat any other failure as retryable with backoff. Most
                # quota / 5xx / network errors from google-genai are
                # retryable; permanent ones will exhaust max_retries and
                # bubble up.
                msg = str(e)
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.cfg.max_retries,
                    msg[:200],
                )
            delay = min(20.0, (2 ** attempt) * 0.5 + random.random() * 0.5)
            await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _image_part(path: Path) -> Any:
        ext = path.suffix.lower().lstrip(".") or "png"
        if ext == "jpg":
            ext = "jpeg"
        with open(path, "rb") as f:
            data = f.read()
        return google_genai_types.Part.from_bytes(
            data=data, mime_type=f"image/{ext}"
        )

    async def caption_chunk(
        self,
        first_frame: Path,
        last_frame: Path,
        mask_image: Path | None,
        middle_frame: Path | None = None,
        *,
        task_description: str = "",
        chunk_index: int = 0,
        num_chunks: int = 1,
        start_frame: int = 0,
        end_frame: int = 0,
        use_motion_mask: bool = True,
    ) -> dict[str, Any]:
        # Gemini accepts a flat list of mixed text and inline image parts as
        # `contents`. We label each image with a leading text part so the
        # model knows the role of the next image. The AgiBot context
        # block (task_description + segment info) is prepended once.
        # When mask use is disabled, switch to the no-mask templates and
        # omit the MOTION_MASK part entirely so the prompt text and
        # image list stay consistent.
        send_mask = use_motion_mask and mask_image is not None
        context_block = build_task_context_block(
            task_description, chunk_index, num_chunks, start_frame, end_frame,
            use_motion_mask=send_mask,
        )
        if middle_frame is not None:
            base_prompt = CHUNK_USER_PROMPT_4IMG if send_mask else CHUNK_USER_PROMPT_3IMG_NOMASK
            user_prompt = context_block + base_prompt
            contents: list[Any] = [
                user_prompt,
                "FIRST_FRAME:",
                self._image_part(first_frame),
                "MIDDLE_FRAME:",
                self._image_part(middle_frame),
                "LAST_FRAME:",
                self._image_part(last_frame),
            ]
        else:
            base_prompt = CHUNK_USER_PROMPT_3IMG if send_mask else CHUNK_USER_PROMPT_2IMG_NOMASK
            user_prompt = context_block + base_prompt
            contents = [
                user_prompt,
                "FIRST_FRAME:",
                self._image_part(first_frame),
                "LAST_FRAME:",
                self._image_part(last_frame),
            ]
        if send_mask:
            contents += ["MOTION_MASK:", self._image_part(mask_image)]
        for json_attempt in range(2):
            text = await self._generate_with_retry(
                contents=contents,
                system_instruction=CHUNK_SYSTEM_PROMPT,
                max_tokens=800,
                model=self.cfg.caption_model,
            )
            try:
                return _extract_json(text)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Gemini malformed JSON on attempt %d: %s",
                    json_attempt + 1,
                    e,
                )
                if json_attempt == 1:
                    return {
                        "instruction_1": "",
                        "instruction_2": "",
                        "instruction_3": "",
                        "embodiment_type": "unknown",
                        "task_type": "manipulation",
                        "skill_labels": ["unknown"],
                        "hallucination_flags": {
                            "contact_claim_without_clear_evidence": False,
                            "object_identity_uncertain": False,
                            "embodiment_uncertain": True,
                        },
                        "structured_motion": {"confidence": "low"},
                        "_error": f"malformed JSON: {e}",
                        "_raw": text,
                    }
        return {}

    async def merge_chunks(
        self,
        chunk_outputs: list[dict[str, Any]],
        *,
        merge_type: str = "",
    ) -> dict[str, Any]:
        prompt = MERGE_USER_PROMPT_TEMPLATE.format(
            chunks_json=json.dumps(chunk_outputs, indent=2)
        )
        # Full merges get a larger budget since they can produce longer
        # JSON. The Gemini path keeps using the configured merge model
        # (no cross-vendor fallback).
        max_tokens = 4096 if merge_type == "full" else 2048
        model = self.cfg.merge_model
        text = await self._generate_with_retry(
            contents=[prompt],
            system_instruction=MERGE_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            model=model,
        )
        try:
            parsed = _extract_json(text)
            logger.info(
                "merge ok via %s (merge_type=%s, n_chunks=%d)",
                model, merge_type or "?", len(chunk_outputs),
            )
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(
                "Gemini merge JSON malformed (model=%s, merge_type=%s, "
                "n_chunks=%d): %s",
                model, merge_type or "?", len(chunk_outputs), e,
            )
            return {
                "instruction_1": "",
                "instruction_2": "",
                "instruction_3": "",
                "timeline": [],
                "_error": f"malformed JSON: {e}",
                "_raw": text,
            }


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------


async def process_chunk(
    client: "CaptionClient | GeminiCaptionClient | None",
    vdata: VideoData,
    video_id: str,
    chunk: Chunk,
    out_root: Path,
    cfg: PipelineConfig,
    *,
    task_description: str = "",
    chunk_index: int = 0,
    num_chunks: int = 1,
) -> dict[str, Any]:
    """Build the mask and (optionally) call the API for one chunk.

    AgiBot extras: receives `task_description` + segment ordinal info
    and threads them into the VLM call. The same fields are also
    written into the chunk record under `agibot_meta` so the training
    export can re-emit them.
    """
    masks_dir = out_root / "masks"
    frames_dir = out_root / "frames"
    captions_dir = out_root / "chunk_captions"
    captions_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    caption_path = captions_dir / f"chunk_{chunk.chunk_id:03d}.json"

    # Check frame availability
    if not vdata.has_frame(chunk.start_idx) or not vdata.has_frame(chunk.end_idx):
        logger.warning(
            "Missing frames for chunk %d (start=%d, end=%d); skipping",
            chunk.chunk_id, chunk.start_idx, chunk.end_idx,
        )
        return {
            "chunk": chunk.to_meta(video_id),
            "agibot_meta": {
                "task_description": task_description,
                "chunk_index": int(chunk_index),
                "num_chunks": int(num_chunks),
                "start_frame": int(chunk.start_idx),
                "end_frame": int(chunk.end_idx),
                "selected_frame_indices": [],
            },
            "error": "missing frames",
        }

    # Ego-motion filtering happens at the frame level *before* chunking
    # (see `process_video`); noisy frames are already excluded from every
    # clean segment, so no per-chunk gate is needed here.
    num_kp_start = vdata.num_keypoints_at(chunk.start_idx)

    try:
        mask = build_motion_mask(vdata, chunk, masks_dir, cfg)
    except Exception as e:
        logger.warning("mask build failed for chunk %d: %s", chunk.chunk_id, e)
        return {
            "chunk": chunk.to_meta(video_id),
            "agibot_meta": {
                "task_description": task_description,
                "chunk_index": int(chunk_index),
                "num_chunks": int(num_chunks),
                "start_frame": int(chunk.start_idx),
                "end_frame": int(chunk.end_idx),
                "selected_frame_indices": [],
            },
            "error": f"mask: {e}",
        }

    # Materialise frame PNGs so the API clients can read them. The RGB
    # lives in images.npy, so we write temp PNGs under curated/frames/.
    first_path = frames_dir / f"chunk_{chunk.chunk_id:03d}_first.png"
    last_path = frames_dir / f"chunk_{chunk.chunk_id:03d}_last.png"
    vdata.save_rgb_to(chunk.start_idx, first_path)
    vdata.save_rgb_to(chunk.end_idx, last_path)

    # Conditional middle frame: include only for long, high-displacement chunks.
    middle_path: Path | None = None
    middle_idx: int | None = None
    use_middle = (
        chunk.length >= cfg.min_frames_for_middle
        and mask.displacement >= cfg.min_disp_for_middle
    )
    if use_middle:
        mid_idx = (chunk.start_idx + chunk.end_idx) // 2
        if vdata.has_frame(mid_idx):
            middle_path = frames_dir / f"chunk_{chunk.chunk_id:03d}_middle.png"
            vdata.save_rgb_to(mid_idx, middle_path)
            middle_idx = int(mid_idx)
            logger.debug(
                "chunk %d: including middle frame %d (len=%d, disp=%.1f)",
                chunk.chunk_id, mid_idx, chunk.length, mask.displacement,
            )

    used_middle_frame = middle_path is not None
    complexity_tier = "medium" if used_middle_frame else "simple"

    selected_frame_indices: list[int] = [int(chunk.start_idx)]
    if middle_idx is not None:
        selected_frame_indices.append(middle_idx)
    selected_frame_indices.append(int(chunk.end_idx))

    agibot_meta = {
        "task_description": task_description,
        "chunk_index": int(chunk_index),
        "num_chunks": int(num_chunks),
        "start_frame": int(chunk.start_idx),
        "end_frame": int(chunk.end_idx),
        "selected_frame_indices": selected_frame_indices,
        "mask_sent_to_vlm": bool(cfg.use_motion_mask),
    }

    record: dict[str, Any] = {
        "chunk": chunk.to_meta(video_id),
        "agibot_meta": agibot_meta,
        "num_keypoints_at_start": int(num_kp_start),
        "used_middle_frame": used_middle_frame,
        "complexity_tier": complexity_tier,
        "frames": {
            "first_frame": str(first_path),
            "last_frame": str(last_path),
            "middle_frame": str(middle_path) if middle_path else None,
            "mask": str(mask.mask_path),
            "overlay": str(mask.overlay_path) if mask.overlay_path else None,
            "mask_cluster_id": mask.cluster_id,
            "mask_n_keypoints": mask.n_keypoints,
            "mask_displacement": mask.displacement,
            "mask_notes": mask.notes,
        },
    }

    if cfg.dry_run or client is None:
        record["caption"] = None
    else:
        if caption_path.exists() and not cfg.overwrite:
            try:
                record["caption"] = json.loads(caption_path.read_text())
                logger.info("reused cached caption for chunk %d", chunk.chunk_id)
                return record
            except Exception:
                pass
        try:
            # Pass the mask path only when the model is actually meant
            # to see it; otherwise pass None so the client cannot
            # accidentally include it in the request.
            caption_mask_path = mask.mask_path if cfg.use_motion_mask else None
            caption = await client.caption_chunk(
                first_path, last_path, caption_mask_path,
                middle_frame=middle_path,
                task_description=task_description,
                chunk_index=chunk_index,
                num_chunks=num_chunks,
                start_frame=int(chunk.start_idx),
                end_frame=int(chunk.end_idx),
                use_motion_mask=cfg.use_motion_mask,
            )
        except Exception as e:
            logger.error("caption call failed for chunk %d: %s", chunk.chunk_id, e)
            caption = {
                "instruction_1": "",
                "instruction_2": "",
                "instruction_3": "",
                "embodiment_type": "unknown",
                "task_type": "manipulation",
                "skill_labels": ["unknown"],
                "hallucination_flags": {
                    "contact_claim_without_clear_evidence": False,
                    "object_identity_uncertain": False,
                    "embodiment_uncertain": True,
                },
                "structured_motion": {"confidence": "low"},
                "_error": str(e),
            }
        caption_path.write_text(json.dumps(caption, indent=2))
        record["caption"] = caption

    return record


# --- Embodiment text pass-through -------------------------------------------
#
# The VLM is now allowed to use natural embodiment wording in text (e.g.
# "the human hand", "the robot gripper") when supported by the visual
# evidence. No post-hoc rewriting is applied — the structured
# `embodiment_type` field carries the canonical label for downstream use.


def _chunk_record_to_merge_input(r: dict[str, Any]) -> dict[str, Any]:
    """Strip a chunk record down to the fields the merge stage actually uses."""
    cap = r.get("caption") or {}
    return {
        "chunk_id": r["chunk"]["chunk_id"],
        "start_idx": r["chunk"]["start_idx"],
        "end_idx": r["chunk"]["end_idx"],
        "instruction_1": cap.get("instruction_1", ""),
        "embodiment_type": cap.get("embodiment_type", "unknown"),
        "skill_labels": cap.get("skill_labels", []),
        "structured_motion": cap.get("structured_motion", {}),
    }


def build_sliding_windows(
    chunk_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build adjacent-only sliding windows *within each clean segment*.

    Chunks are first grouped by their ``segment_id`` (assigned during
    segment-wise chunking). Windows (pairs, triples, full) are then
    generated **only** over chunks that share a segment, so merges never
    cross a noisy gap. Within a segment the behaviour matches the
    original: pairs, triples, and one full-segment window; duplicates
    (e.g. pair == full on n=2) are deduplicated.

    ``source_chunk_ids`` preserves global chunk ids so the training
    export and cache keys remain stable.
    """
    out: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, ...]] = set()

    def _add(merge_type: str, slice_records: list[dict[str, Any]]) -> None:
        if not slice_records:
            return
        ids = tuple(r["chunk"]["chunk_id"] for r in slice_records)
        if ids in seen_keys:
            return
        seen_keys.add(ids)
        out.append(
            {
                "merge_type": merge_type,
                "source_chunk_ids": list(ids),
                "segment_id": int(slice_records[0]["chunk"].get("segment_id", 0)),
                "start_idx": int(slice_records[0]["chunk"]["start_idx"]),
                "end_idx": int(slice_records[-1]["chunk"]["end_idx"]),
                "records": slice_records,
            }
        )

    # Group by segment_id, preserving chunk order within each segment.
    groups: dict[int, list[dict[str, Any]]] = {}
    for r in chunk_records:
        sid = int(r["chunk"].get("segment_id", 0))
        groups.setdefault(sid, []).append(r)

    for sid in sorted(groups):
        group = groups[sid]
        n = len(group)
        if n >= 2:
            for i in range(n - 1):
                _add("pair", group[i : i + 2])
        if n >= 3:
            for i in range(n - 2):
                _add("triple", group[i : i + 3])
        if n >= 1:
            _add("full", group[:])
    return out


async def _run_one_merge(
    client: "CaptionClient | GeminiCaptionClient",
    window: dict[str, Any],
) -> dict[str, Any]:
    """Run a single merge call for one sliding window.

    Concurrency note: every merge call goes through the same client
    instance and therefore the SAME `client.semaphore`. The semaphore
    bounds total in-flight requests at `cfg.max_concurrent_requests`, so
    chunk caption calls and merge calls share one global request budget
    and can never collectively exceed the Gemini rate limit.
    """
    # Single-chunk window: nothing to merge, so skip the API call and
    # synthesize the merged record directly from the lone chunk's caption.
    if len(window["records"]) == 1:
        rec = window["records"][0]
        chunk = rec["chunk"]
        cap = rec.get("caption") or {}
        instr1 = cap.get("instruction_1", "")
        return {
            "merge_type": window["merge_type"],
            "source_chunk_ids": window["source_chunk_ids"],
            "start_idx": window["start_idx"],
            "end_idx": window["end_idx"],
            "instruction_1": instr1,
            "instruction_2": cap.get("instruction_2", ""),
            "instruction_3": cap.get("instruction_3", ""),
            "skill_labels": cap.get("skill_labels", []),
            "timeline": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "start_idx": chunk["start_idx"],
                    "end_idx": chunk["end_idx"],
                    "event": instr1,
                }
            ],
        }

    merge_input = [_chunk_record_to_merge_input(r) for r in window["records"]]
    try:
        merged = await client.merge_chunks(
            merge_input, merge_type=window["merge_type"]
        )
    except Exception as e:
        logger.error(
            "merge call failed (%s, chunks=%s): %s",
            window["merge_type"],
            window["source_chunk_ids"],
            e,
        )
        merged = {
            "instruction_1": "",
            "instruction_2": "",
            "instruction_3": "",
            "timeline": [],
            "_error": str(e),
        }
    return {
        "merge_type": window["merge_type"],
        "source_chunk_ids": window["source_chunk_ids"],
        "start_idx": window["start_idx"],
        "end_idx": window["end_idx"],
        **merged,
    }


def _merge_cache_key(merged_record: dict[str, Any]) -> str:
    return f"{merged_record['merge_type']}:{'-'.join(map(str, merged_record['source_chunk_ids']))}"


def _build_training_export(
    video_id: str,
    n_frames: int,
    chunk_records: list[dict[str, Any]],
    merged_records: list[dict[str, Any]],
    *,
    task_description: str = "",
) -> dict[str, Any]:
    """Lean training-oriented export.

    Keeps text + frame ranges + the new semantic labels needed for
    training:
      * per-chunk: text, frame range, embodiment_type, skill_labels,
        hallucination_flags, used_middle_frame, complexity_tier, plus
        the AgiBot extras (task_description, chunk_index, num_chunks,
        start_frame, end_frame, selected_frame_indices)
      * per-merged-window: text, merge_type, source chunk ids, frame
        coverage, aggregated skill_labels
    """
    chunk_texts: list[dict[str, Any]] = []
    for r in chunk_records:
        cap = r.get("caption") or {}
        meta = r.get("agibot_meta") or {}
        entry: dict[str, Any] = {
            "chunk_id": r["chunk"]["chunk_id"],
            "start_idx": r["chunk"]["start_idx"],
            "end_idx": r["chunk"]["end_idx"],
            "instruction_1": cap.get("instruction_1", ""),
            "instruction_2": cap.get("instruction_2", ""),
            "instruction_3": cap.get("instruction_3", ""),
            "embodiment_type": cap.get("embodiment_type", "unknown"),
            "task_type": cap.get("task_type", "manipulation"),
            "skill_labels": cap.get("skill_labels", []),
            "hallucination_flags": cap.get("hallucination_flags", {}),
            "used_middle_frame": r.get("used_middle_frame", False),
            "complexity_tier": r.get("complexity_tier", "simple"),
            # AgiBot per-chunk extras
            "task_description": meta.get("task_description", task_description),
            "chunk_index": meta.get("chunk_index", r["chunk"]["chunk_id"]),
            "num_chunks": meta.get("num_chunks", len(chunk_records)),
            "start_frame": meta.get("start_frame", r["chunk"]["start_idx"]),
            "end_frame": meta.get("end_frame", r["chunk"]["end_idx"]),
            "selected_frame_indices": meta.get("selected_frame_indices", []),
        }
        if r.get("error") or "_error" in cap:
            entry["error"] = r.get("error") or cap.get("_error")
        chunk_texts.append(entry)

    merged_texts: list[dict[str, Any]] = []
    for m in merged_records:
        timeline = m.get("timeline") or []
        merged_texts.append(
            {
                "merge_type": m["merge_type"],
                "source_chunk_ids": m["source_chunk_ids"],
                "start_idx": m["start_idx"],
                "end_idx": m["end_idx"],
                "instruction_1": m.get("instruction_1", ""),
                "instruction_2": m.get("instruction_2", ""),
                "instruction_3": m.get("instruction_3", ""),
                "skill_labels": m.get("skill_labels", []),
                "timeline": timeline,
            }
        )

    return {
        "video_id": video_id,
        "n_frames": int(n_frames),
        "task_description": task_description,
        "chunk_texts": chunk_texts,
        "merged_texts": merged_texts,
    }


def _warn_on_duplicate_instructions(
    video_id: str,
    chunk_records: list[dict[str, Any]],
) -> None:
    """Log a WARNING when adjacent chunks have identical `instruction_1`.

    Identical adjacent captions almost always mean the task prior is
    overpowering segment-specific visual grounding — every chunk
    collapses onto the same generic task-level sentence. The merge
    stage compounds this, so it is worth flagging explicitly.
    """
    def _norm(s: str) -> str:
        return " ".join((s or "").lower().split())

    n_dups = 0
    for prev, curr in zip(chunk_records, chunk_records[1:]):
        prev_cap = (prev.get("caption") or {}).get("instruction_1", "")
        curr_cap = (curr.get("caption") or {}).get("instruction_1", "")
        if not prev_cap or not curr_cap:
            continue
        if _norm(prev_cap) == _norm(curr_cap):
            n_dups += 1
            logger.warning(
                "%s: chunks %d and %d share identical instruction_1 "
                "(%r). The task prior may be overpowering segment-"
                "specific visual grounding.",
                video_id,
                prev["chunk"]["chunk_id"],
                curr["chunk"]["chunk_id"],
                prev_cap[:160],
            )
    if n_dups:
        logger.warning(
            "%s: %d adjacent-chunk duplicate(s) detected in instruction_1; "
            "merge results may be unreliable.",
            video_id, n_dups,
        )


async def process_video(
    client: "CaptionClient | GeminiCaptionClient | None",
    video_dir: Path,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    video_id = episode_id_from_dir(video_dir)
    logger.info("=== %s ===", video_id)

    out_root = video_dir / "curated"
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "%s: VLM motion-mask is %s",
        video_id,
        "ENABLED (sent to model)" if cfg.use_motion_mask else "DISABLED",
    )

    # AgiBot: load the per-episode task description once so it can be
    # threaded into every chunk's caption call. Empty string is a valid
    # value (helper logs a warning + returns "" for missing/unreadable).
    task_description = read_task_description(video_dir)
    if task_description:
        logger.info(
            "%s: loaded task_description (%d chars): %r",
            video_id, len(task_description), task_description[:120],
        )
    else:
        logger.info("%s: no description.txt; task_description is empty", video_id)

    # Open per-video data handles once — reused across all chunks.
    vdata = VideoData(video_dir, video_id)

    accel_path = video_dir / "acceleration.npz"
    with np.load(accel_path) as f:
        if "acceleration" not in f.files:
            raise KeyError(f"{accel_path} does not contain 'acceleration'")
        acc = np.asarray(f["acceleration"], dtype=np.float32)

    # Cross-check against actual frame count.
    n_frames = vdata.n_frames
    if n_frames != len(acc):
        logger.warning(
            "%s: frames=%d but acceleration length=%d — using min",
            video_id,
            n_frames,
            len(acc),
        )
        acc = acc[: min(n_frames, len(acc))]

    # ------------------------------------------------------------------
    # Step 1 — frame-level ego-motion filter, THEN per-segment chunking.
    # The noisy frames are excluded *before* smoothing + peak detection
    # so the big ego-motion acceleration spikes cannot corrupt peaks on
    # the clean portions of the timeline.
    # ------------------------------------------------------------------
    per_frame_counts = vdata.per_frame_keypoint_counts()[: len(acc)]
    noisy_intervals = find_noisy_intervals(
        per_frame_counts, cfg.max_keypoints_per_frame, pad=cfg.noisy_pad
    )
    effective_min_len = min(cfg.min_chunk_len, max(2, len(acc)))
    segments = clean_segments_from_noisy(
        len(acc), noisy_intervals, min_len=effective_min_len
    )
    logger.info(
        "%s: frame filter → %d noisy interval(s), %d clean segment(s)",
        video_id, len(noisy_intervals), len(segments),
    )

    chunks: list[Chunk] = []
    next_id = 0
    for seg_id, (s, e) in enumerate(segments):
        seg_chunks = chunk_acceleration(acc, cfg, segment_start=s, segment_end=e)
        for c in seg_chunks:
            c.chunk_id = next_id
            c.segment_id = seg_id
            next_id += 1
        chunks.extend(seg_chunks)
    logger.info(
        "%s: %d chunks across %d clean segment(s)",
        video_id, len(chunks), len(segments),
    )

    chunks_meta = [c.to_meta(video_id) for c in chunks]
    (out_root / "chunks_meta.json").write_text(json.dumps(chunks_meta, indent=2))
    (out_root / "segments_meta.json").write_text(json.dumps({
        "n_frames": int(len(acc)),
        "max_keypoints_per_frame": int(cfg.max_keypoints_per_frame),
        "noisy_pad": int(cfg.noisy_pad),
        "noisy_intervals": [
            {"start_idx": int(s), "end_idx": int(e), "length": int(e - s + 1)}
            for s, e in noisy_intervals
        ],
        "clean_segments": [
            {"segment_id": i, "start_idx": int(s), "end_idx": int(e),
             "length": int(e - s + 1)}
            for i, (s, e) in enumerate(segments)
        ],
    }, indent=2))

    # Step 2/3 — chunk-level multimodal captioning. The CaptionClient
    # semaphore caps in-flight requests at cfg.max_concurrent_requests,
    # so even on a long video we never blow past the rate limit.
    num_chunks = len(chunks)
    tasks = [
        process_chunk(
            client, vdata, video_id, c, out_root, cfg,
            task_description=task_description,
            chunk_index=i,
            num_chunks=num_chunks,
        )
        for i, c in enumerate(chunks)
    ]
    chunk_records: list[dict[str, Any]] = await asyncio.gather(*tasks)

    # AgiBot diagnostic: warn when adjacent chunks share an identical
    # `instruction_1`. This usually means the task prior is overpowering
    # segment-specific visual grounding (often producing the same
    # generic task-level sentence for every chunk).
    _warn_on_duplicate_instructions(video_id, chunk_records)

    # Step 4 — sliding-window merges (text only). Pairs / triples / full,
    # all running concurrently and sharing the same client semaphore so
    # the merge stage cannot overflow the Gemini rate budget either.
    merged_records: list[dict[str, Any]] = []
    if not cfg.dry_run and client is not None:
        windows = build_sliding_windows(chunk_records)
        logger.info(
            "%s: scheduling %d sliding-window merges", video_id, len(windows)
        )

        # Caching: load any existing merged_windows.json so a re-run only
        # re-issues calls for windows that were not previously cached.
        merged_windows_path = out_root / "merged_windows.json"
        cached: dict[str, dict[str, Any]] = {}
        if merged_windows_path.exists() and not cfg.overwrite:
            try:
                prior = json.loads(merged_windows_path.read_text())
                for rec in prior:
                    cached[_merge_cache_key(rec)] = rec
            except Exception as e:
                logger.warning(
                    "could not parse cached %s: %s", merged_windows_path, e
                )

        async def _maybe_run(window: dict[str, Any]) -> dict[str, Any]:
            key_rec = {
                "merge_type": window["merge_type"],
                "source_chunk_ids": window["source_chunk_ids"],
            }
            key = _merge_cache_key(key_rec)
            if key in cached:
                logger.debug("reusing cached merge %s", key)
                return cached[key]
            return await _run_one_merge(client, window)

        merged_records = await asyncio.gather(*[_maybe_run(w) for w in windows])
        merged_windows_path.write_text(json.dumps(merged_records, indent=2))

    # Intermediate / debug record — full chunk caption JSON + raw merged
    # windows. Useful for inspection and re-runs; not the training file.
    final = {
        "video_id": video_id,
        "video_dir": str(video_dir),
        "n_frames": int(len(acc)),
        "task_description": task_description,
        "config": {
            "backend": cfg.backend,
            "caption_model": cfg.caption_model,
            "merge_model": cfg.merge_model,
            "smooth_window": cfg.smooth_window,
            "min_chunk_len": cfg.min_chunk_len,
            "max_chunk_len": cfg.max_chunk_len,
            "accel_threshold_ratio": cfg.accel_threshold_ratio,
            "min_prominence_ratio": cfg.min_prominence_ratio,
            "max_chunks": cfg.max_chunks,
            "use_motion_mask": cfg.use_motion_mask,
        },
        "chunks": chunk_records,
        "merged_windows": merged_records,
    }
    if cfg.dry_run:
        # Placeholder (empty) captions must not land in the final files: they
        # would be read by training and mark the episode as done for re-runs.
        logger.info("%s: dry run, not writing final caption files", video_id)
        return final

    (video_dir / "curated_captions.json").write_text(json.dumps(final, indent=2))

    # Lean training-oriented export — only caption text + frame ranges.
    training = _build_training_export(
        video_id, len(acc), chunk_records, merged_records,
        task_description=task_description,
    )
    (video_dir / "curated_training_texts.json").write_text(json.dumps(training, indent=2))

    logger.info(
        "%s: wrote %s and %s",
        video_id,
        video_dir / "curated_captions.json",
        video_dir / "curated_training_texts.json",
    )
    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Curate motion captions for a folder of episodes."
    )
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root containing episode_* folders.",
    )
    p.add_argument(
        "--video",
        action="append",
        default=None,
        help="Process only this episode id (may be repeated).",
    )
    p.add_argument(
        "--shard",
        type=int,
        default=None,
        help=(
            "1-indexed shard id. When set, process only episodes "
            "[(shard-1)*shard_size : shard*shard_size] from the "
            "deterministically sorted episode list. Applied after "
            "--video filtering."
        ),
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of episodes per shard (default: 10000).",
    )
    p.add_argument(
        "--backend",
        choices=["openai", "gemini"],
        default="openai",
        help="Which captioning backend to use.",
    )
    p.add_argument(
        "--caption-model",
        default=None,
        help=(
            "Model used for chunk-level multimodal captioning. "
            "Defaults to OPENAI_CAPTION_MODEL / GEMINI_CAPTION_MODEL "
            "from .env, falling back to gpt-4o-mini / gemini-2.5-flash."
        ),
    )
    p.add_argument(
        "--merge-model",
        default=None,
        help=(
            "Model used for the text-only merge stage. "
            "Defaults to OPENAI_MERGE_MODEL / GEMINI_MERGE_MODEL "
            "from .env, falling back to gpt-4o-mini / "
            "gemini-2.5-flash-lite."
        ),
    )
    p.add_argument("--smooth-window", type=int, default=11)
    p.add_argument("--min-chunk-len", type=int, default=25,
                    help="Minimum frames per chunk. AgiBot default is 25 "
                         "(was 8 in DROID curate_captions.py) so that "
                         "tiny sub-chunks get merged into a neighbour.")
    p.add_argument("--max-chunk-len", type=int, default=300,
                    help="Soft safety guardrail on chunk length. AgiBot "
                         "default is 300; an episode is only force-split "
                         "if a single chunk exceeds this.")
    p.add_argument("--accel-threshold-ratio", type=float, default=0.35)
    p.add_argument("--min-prominence-ratio", type=float, default=0.40,
                    help="Minimum peak prominence as fraction of accel "
                         "range. AgiBot default 0.40 keeps only major "
                         "motion events as chunk-boundary candidates.")
    p.add_argument("--max-chunks", type=int, default=3,
                    help="Maximum number of chunks per video. AgiBot "
                         "default is 3 (episodes are already task-level).")
    p.add_argument("--pad-frames", type=int, default=1)
    p.add_argument("--mask-radius", type=int, default=12)
    p.add_argument("--mask-dilate", type=int, default=5)
    p.add_argument(
        "--use-motion-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether to send the motion-mask image to the VLM. AgiBot "
            "default is False because the mask tends to over-focus the "
            "model on low-level arm motion at the expense of "
            "task-relevant object state changes. Pass "
            "--use-motion-mask to re-enable; pass --no-use-motion-mask "
            "to force-disable. Mask artefacts are still written under "
            "`curated/masks/` either way."
        ),
    )
    p.add_argument("--min-frames-for-middle", type=int, default=30,
                    help="Minimum chunk length to include a middle frame.")
    p.add_argument("--min-disp-for-middle", type=float, default=5.0,
                    help="Minimum displacement (px) to include a middle frame.")
    p.add_argument("--max-keypoints-per-frame", type=int, default=300,
                    help="Mark any frame with >= this many keypoints as "
                         "noisy (ego-motion / global-motion filter). Noisy "
                         "frames are excluded *before* chunking.")
    p.add_argument("--noisy-pad", type=int, default=0,
                    help="Expand each noisy interval by this many frames "
                         "on each side before building clean segments.")
    p.add_argument("--max-concurrent-requests", type=int, default=6)
    p.add_argument("--request-timeout", type=float, default=60.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run chunking + masks but skip every API call.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run captioning even if cached chunk JSON exists.",
    )
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    if args.backend == "gemini":
        default_caption = os.environ.get(
            "GEMINI_CAPTION_MODEL",
            os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        )
        default_merge = os.environ.get(
            "GEMINI_MERGE_MODEL",
            os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        )
    else:
        default_caption = os.environ.get(
            "OPENAI_CAPTION_MODEL",
            os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        )
        default_merge = os.environ.get("OPENAI_MERGE_MODEL", "gpt-4o-mini")
    caption_model = args.caption_model or default_caption
    merge_model = args.merge_model or default_merge
    return PipelineConfig(
        root=args.root,
        backend=args.backend,
        caption_model=caption_model,
        merge_model=merge_model,
        smooth_window=args.smooth_window,
        min_chunk_len=args.min_chunk_len,
        max_chunk_len=args.max_chunk_len,
        accel_threshold_ratio=args.accel_threshold_ratio,
        min_prominence_ratio=args.min_prominence_ratio,
        max_chunks=args.max_chunks,
        pad_frames=args.pad_frames,
        mask_radius=args.mask_radius,
        mask_dilate=args.mask_dilate,
        use_motion_mask=args.use_motion_mask,
        min_frames_for_middle=args.min_frames_for_middle,
        min_disp_for_middle=args.min_disp_for_middle,
        max_keypoints_per_frame=args.max_keypoints_per_frame,
        noisy_pad=args.noisy_pad,
        max_concurrent_requests=args.max_concurrent_requests,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


async def amain(args: argparse.Namespace) -> int:
    cfg = build_config(args)
    if not cfg.root.exists():
        logger.error("dataset root %s does not exist", cfg.root)
        return 2

    if args.shard is not None and args.shard <= 0:
        logger.error("--shard must be a positive integer (got %d)", args.shard)
        return 2
    if args.shard_size <= 0:
        logger.error(
            "--shard-size must be a positive integer (got %d)", args.shard_size
        )
        return 2

    videos = discover_videos(cfg.root)
    logger.info("Discovered %d episodes under %s", len(videos), cfg.root)
    if args.video:
        wanted = set(args.video)
        videos = [v for v in videos if v.name in wanted]
        logger.info(
            "Applied --video filter: %d episodes remain after filtering",
            len(videos),
        )
    if not videos:
        logger.error("no episodes found under %s", cfg.root)
        return 2

    if args.shard is not None:
        total_after_filter = len(videos)
        start = (args.shard - 1) * args.shard_size
        end = args.shard * args.shard_size
        if start >= total_after_filter:
            logger.error(
                "shard start index %d is beyond the number of available "
                "episodes (%d); nothing to do for shard %d with "
                "shard_size=%d",
                start,
                total_after_filter,
                args.shard,
                args.shard_size,
            )
            return 2
        videos = videos[start:end]
        actual_end = start + len(videos)
        logger.info(
            "Using shard %d with shard_size=%d: selected episodes "
            "%d-%d (%d episodes)%s",
            args.shard,
            args.shard_size,
            start + 1,
            actual_end,
            len(videos),
            " (after --video filter)" if args.video else "",
        )

    if not cfg.overwrite:
        pending: list[Path] = []
        for v in videos:
            if episode_has_completed_outputs(v):
                logger.info("skipping %s (already processed)", episode_id_from_dir(v))
            else:
                pending.append(v)
        videos = pending
        if not videos:
            logger.info("no episodes to process")
            return 0

    client: CaptionClient | GeminiCaptionClient | None = None
    if not cfg.dry_run:
        if cfg.backend == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                logger.error(
                    "OPENAI_API_KEY is not set; either fix .env or use --dry-run"
                )
                return 2
            client = CaptionClient(cfg)
        elif cfg.backend == "gemini":
            if not (
                os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            ):
                logger.error(
                    "GEMINI_API_KEY is not set; either fix .env or use --dry-run"
                )
                return 2
            client = GeminiCaptionClient(cfg)
        else:
            logger.error("unknown backend: %s", cfg.backend)
            return 2

    t0 = time.time()
    for v in videos:
        try:
            await process_video(client, v, cfg)
        except Exception as e:
            logger.exception("video %s failed: %s", v.name, e)
    logger.info("done in %.1fs", time.time() - t0)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    # Load .env early so API keys and model overrides are available.
    load_dotenv()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
