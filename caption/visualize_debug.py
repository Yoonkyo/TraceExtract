#!/usr/bin/env python3
"""
visualize_debug.py
==================

Generate an MP4 video for a processed episode that overlays keypoint
trajectories and chunk captions on top of the RGB frames.  Useful for
visually debugging whether:

  1. The chunking boundaries are reasonable.
  2. The captions align with the actual motion.
  3. The dominant-cluster trace follows the moving region.

Usage examples
--------------
# Basic — red future trajectories only
python visualize_debug.py outputs/droid/droid_shard00000_ep000

# Red future + green history trajectories
python visualize_debug.py outputs/droid/droid_shard00000_ep000 --visualize_history

# Colour-code trajectories by cluster ID
python visualize_debug.py outputs/droid/droid_shard00000_ep000 --classify_cluster

# Colour-code trajectories by cluster ID + history (dashed)
python visualize_debug.py outputs/droid/droid_shard00000_ep000 --classify_cluster --visualize_history

# Choose which caption field to display
python visualize_debug.py outputs/droid/droid_shard00000_ep000 --caption-field instruction_2

# Tweak trajectory horizon and FPS
python visualize_debug.py outputs/droid/droid_shard00000_ep000 --horizon 16 --fps 10

# Process all episodes under a root
python visualize_debug.py outputs/droid --all-videos
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required.  pip install opencv-python", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_video_data(video_dir: Path, load_history: bool = False) -> dict:
    """Load the unified .npy arrays for one video directory."""
    images = np.load(video_dir / "images.npy", mmap_mode="r")
    samples = video_dir / "samples"
    frame_indices = np.load(samples / "frame_indices.npy")
    offsets = np.load(samples / "offsets.npy")
    keypoints = np.load(samples / "keypoints.npy", mmap_mode="r")
    cluster_ids = np.load(samples / "cluster_ids.npy", mmap_mode="r")
    raw_traj = np.load(samples / "raw_traj.npy", mmap_mode="r")
    raw_valid = np.load(samples / "raw_valid_steps.npy", mmap_mode="r")
    frame_to_slot = {int(t): i for i, t in enumerate(frame_indices)}
    result = {
        "images": images,
        "frame_indices": frame_indices,
        "offsets": offsets,
        "keypoints": keypoints,
        "cluster_ids": cluster_ids,
        "raw_traj": raw_traj,
        "raw_valid": raw_valid,
        "frame_to_slot": frame_to_slot,
        "n_frames": int(images.shape[0]),
    }
    if load_history:
        hist_traj_path = samples / "raw_traj_history.npy"
        hist_valid_path = samples / "raw_valid_steps_history.npy"
        if hist_traj_path.exists() and hist_valid_path.exists():
            result["raw_traj_history"] = np.load(hist_traj_path, mmap_mode="r")
            result["raw_valid_history"] = np.load(hist_valid_path, mmap_mode="r")
        else:
            print(f"WARNING: history files not found in {samples}, skipping history", file=sys.stderr)
    return result


def get_frame_sample(vd: dict, t: int) -> dict | None:
    """Slice the concatenated arrays for frame *t*, or return None."""
    slot = vd["frame_to_slot"].get(t)
    if slot is None:
        return None
    lo = int(vd["offsets"][slot])
    hi = int(vd["offsets"][slot + 1])
    if hi <= lo:
        return None
    result = {
        "keypoints": np.array(vd["keypoints"][lo:hi], dtype=np.float32),
        "cluster_ids": np.array(vd["cluster_ids"][lo:hi], dtype=np.int32),
        "raw_traj": np.array(vd["raw_traj"][lo:hi], dtype=np.float32),
        "raw_valid": np.array(vd["raw_valid"][lo:hi]),
    }
    if "raw_traj_history" in vd:
        result["raw_traj_history"] = np.array(vd["raw_traj_history"][lo:hi], dtype=np.float32)
        result["raw_valid_history"] = np.array(vd["raw_valid_history"][lo:hi])
    return result


def load_captions(video_dir: Path) -> dict:
    """Load curated_training_texts.json and build frame→chunk lookup.

    Supports two schemas:
      • legacy: {"chunks": [{"chunk": {...}, "caption": {...}}, ...]}
      • training texts: {"chunk_texts": [{"chunk_id", "start_idx", "end_idx",
                          "instruction_1", ...}, ...]}
    Records are normalized to the legacy nested form so the drawing code
    can stay schema-agnostic.
    """
    captions_path = video_dir / "curated_training_texts.json"
    if not captions_path.exists():
        return {"chunks": [], "frame_to_chunk": {}}
    data = json.loads(captions_path.read_text())

    raw_chunks = data.get("chunks") or data.get("chunk_texts") or []
    chunks: list[dict] = []
    for rec in raw_chunks:
        if "chunk" in rec and "caption" in rec:
            chunks.append(rec)
            continue
        chunk_meta = {
            "chunk_id": rec.get("chunk_id"),
            "start_idx": rec.get("start_idx", 0),
            "end_idx": rec.get("end_idx", 0),
        }
        caption = {k: v for k, v in rec.items()
                   if k not in ("chunk_id", "start_idx", "end_idx")}
        chunks.append({"chunk": chunk_meta, "caption": caption})

    frame_to_chunk: dict[int, dict] = {}
    for rec in chunks:
        c = rec["chunk"]
        s, e = c.get("start_idx", 0), c.get("end_idx", 0)
        for t in range(s, e + 1):
            frame_to_chunk[t] = rec
    return {"chunks": chunks, "frame_to_chunk": frame_to_chunk, "raw": data}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


# Per-cluster color palette (BGR for OpenCV).  20 distinct colours from
# matplotlib tab20, pre-converted to BGR uint8.
_PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]
_PALETTE_BGR = [(b, g, r) for (r, g, b) in _PALETTE]


def _cluster_color(cid: int) -> tuple[int, int, int]:
    return _PALETTE_BGR[cid % len(_PALETTE_BGR)]


_COLOR_RED_BGR = (0, 0, 255)
_COLOR_GREEN_BGR = (0, 200, 0)


def draw_keypoints(
    canvas: np.ndarray,
    sample: dict,
    radius: int = 3,
    scale: float = 1.0,
    classify_cluster: bool = True,
    color_override: tuple[int, int, int] | None = None,
) -> None:
    """Draw keypoint dots on the canvas."""
    kp = sample["keypoints"]
    cids = sample["cluster_ids"]
    for i in range(len(kp)):
        x, y = int(round(kp[i, 0] * scale)), int(round(kp[i, 1] * scale))
        cid = int(cids[i])
        if color_override is not None:
            color = color_override
        elif classify_cluster:
            color = _cluster_color(cid)
        else:
            color = _COLOR_RED_BGR
        cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)


def _draw_dashed_polyline(
    canvas: np.ndarray,
    pts: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 8,
    gap_len: int = 6,
) -> None:
    """Draw a dashed polyline by walking along consecutive point pairs."""
    for seg_idx in range(len(pts) - 1):
        x0, y0 = pts[seg_idx]
        x1, y1 = pts[seg_idx + 1]
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1:
            continue
        ux, uy = dx / length, dy / length
        drawn = 0.0
        drawing = True
        while drawn < length:
            step = dash_len if drawing else gap_len
            end = min(drawn + step, length)
            if drawing:
                sx = int(round(x0 + ux * drawn))
                sy = int(round(y0 + uy * drawn))
                ex = int(round(x0 + ux * end))
                ey = int(round(y0 + uy * end))
                cv2.line(canvas, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
            drawn = end
            drawing = not drawing


def draw_trajectories(
    canvas: np.ndarray,
    sample: dict,
    horizon: int = 8,
    scale: float = 1.0,
    classify_cluster: bool = True,
    color_override: tuple[int, int, int] | None = None,
    traj_key: str = "raw_traj",
    valid_key: str = "raw_valid",
    dashed: bool = False,
) -> None:
    """Draw trajectory lines for each keypoint.

    When *classify_cluster* is True the colour comes from the cluster palette;
    otherwise a fixed *color_override* (default red) is used.
    Set *dashed* to True to render as a dashed line (used for history).
    """
    kp = sample["keypoints"]
    cids = sample["cluster_ids"]
    traj = sample[traj_key]           # (K, T, 3)
    valid = sample[valid_key]         # (K, T)
    K = len(kp)
    for i in range(K):
        cid = int(cids[i])
        if color_override is not None:
            color = color_override
        elif classify_cluster:
            color = _cluster_color(cid)
        else:
            color = _COLOR_RED_BGR
        track_valid = valid[i].astype(bool)
        pts_raw = traj[i]  # (T, 3)
        pts_2d: list[tuple[int, int]] = []
        for j in range(min(horizon, pts_raw.shape[0])):
            if not track_valid[j]:
                continue
            px, py = float(pts_raw[j, 0]), float(pts_raw[j, 1])
            if not (np.isfinite(px) and np.isfinite(py)):
                continue
            pts_2d.append((int(round(px * scale)), int(round(py * scale))))
        if len(pts_2d) < 2:
            continue
        if dashed:
            _draw_dashed_polyline(canvas, pts_2d, color, thickness=1)
        else:
            arr = np.array(pts_2d, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [arr], False, color, 1, cv2.LINE_AA)
        ex, ey = pts_2d[-1]
        cv2.drawMarker(canvas, (ex, ey), color, cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)


def draw_chunk_boundary(
    canvas: np.ndarray,
    t: int,
    chunk_rec: dict | None,
    prev_chunk_rec: dict | None,
) -> None:
    """Draw a coloured left-edge bar when a new chunk starts."""
    if chunk_rec is None:
        return
    cid = chunk_rec.get("chunk", {}).get("chunk_id")
    if cid is None:
        return
    # Only draw at the start frame of a chunk
    s = chunk_rec["chunk"].get("start_idx", -1)
    if t != s:
        return
    h = canvas.shape[0]
    color = _cluster_color(cid)
    cv2.rectangle(canvas, (0, 0), (6, h), color, -1)


def wrap_text(text: str, max_chars: int = 60) -> list[str]:
    """Word-wrap a string into lines of at most *max_chars* characters."""
    return textwrap.wrap(text, width=max_chars) or [""]


_CAPTION_FALLBACK_CHAIN = [
    "instruction_1", "instruction_2", "instruction_3",
    "short_description", "detailed_description",
]


def _resolve_caption_text(cap: dict, field: str) -> str:
    """Try *field* first, then walk a fallback chain for older schemas."""
    text = cap.get(field, "")
    if text:
        return text
    for alt in _CAPTION_FALLBACK_CHAIN:
        if alt != field:
            text = cap.get(alt, "")
            if text:
                return text
    return ""


def draw_caption(
    canvas: np.ndarray,
    chunk_rec: dict | None,
    caption_field: str,
) -> None:
    """Overlay chunk caption text at the bottom of the frame.

    Font size, line height, and wrapping all scale with the canvas
    height so the caption stays readable on both small (360p droid)
    and large (720p egoverse) frames without clipping.
    """
    h, w = canvas.shape[:2]
    # Scale everything relative to a 720p reference height.
    ref_h = 720.0
    s = h / ref_h
    font_scale = max(0.45, 0.9 * s)
    thickness = max(1, round(1.8 * s))
    line_h = max(20, int(34 * s))
    pad = max(4, int(8 * s))

    if chunk_rec is None:
        text = "(no chunk data)"
    else:
        cap = chunk_rec.get("caption") or {}
        cid = chunk_rec.get("chunk", {}).get("chunk_id", "?")
        start = chunk_rec.get("chunk", {}).get("start_idx", "?")
        end = chunk_rec.get("chunk", {}).get("end_idx", "?")
        text = _resolve_caption_text(cap, caption_field)
        if not text:
            text = "(no caption text available)"
        text = f"[chunk {cid} | {start}-{end}] {text}"

    # Estimate chars per line using average character width of a representative
    # sample string instead of a single wide letter like "A", which over-
    # estimates and causes premature line wrapping.
    font = cv2.FONT_HERSHEY_SIMPLEX
    _sample = "The quick brown fox jumps over the lazy dog 0123456789"
    (sample_w, _), _ = cv2.getTextSize(_sample, font, font_scale, thickness)
    char_w = sample_w / len(_sample)
    chars_per_line = max(20, int((w - 2 * pad) / max(char_w, 1)))
    lines = wrap_text(text, max_chars=chars_per_line)
    # Cap to at most 4 lines so the box never covers too much of the frame.
    if len(lines) > 4:
        lines = lines[:4]
        lines[-1] = lines[-1][:max(0, len(lines[-1]) - 3)] + "..."

    box_h = len(lines) * line_h + 2 * pad
    # Semi-transparent background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h - box_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
    for i, line in enumerate(lines):
        y = h - box_h + pad + (i + 1) * line_h - 4
        cv2.putText(canvas, line, (pad, y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)


def draw_hud(canvas: np.ndarray, t: int, n_frames: int, n_kp: int) -> None:
    """Heads-up display in the top-left corner, scaled to frame size."""
    h = canvas.shape[0]
    s = h / 720.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, 0.7 * s)
    thickness = max(1, round(1.4 * s))
    y = max(18, int(28 * s))
    text = f"frame {t}/{n_frames - 1}  kp={n_kp}"
    cv2.putText(canvas, text, (8, y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (8, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------


def generate_video(
    video_dir: Path,
    output_path: Path,
    caption_field: str = "instruction_1",
    horizon: int = 8,
    fps: int = 5,
    scale: float = 1.0,
    classify_cluster: bool = False,
    visualize_history: bool = False,
) -> None:
    """Generate the debug MP4 for one episode directory."""
    print(f"Loading data from {video_dir} ...")
    vd = load_video_data(video_dir, load_history=visualize_history)
    cap_data = load_captions(video_dir)
    n_frames = vd["n_frames"]
    h0, w0 = int(vd["images"].shape[1]), int(vd["images"].shape[2])
    h_out = max(1, int(h0 * scale))
    w_out = max(1, int(w0 * scale))
    has_history = "raw_traj_history" in vd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w_out, h_out))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")

    prev_chunk_rec = None
    for t in range(n_frames):
        rgb = np.array(vd["images"][t])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if scale != 1.0:
            bgr = cv2.resize(bgr, (w_out, h_out), interpolation=cv2.INTER_LINEAR)

        sample = get_frame_sample(vd, t)
        chunk_rec = cap_data["frame_to_chunk"].get(t)

        if sample is not None:
            if visualize_history and has_history and "raw_traj_history" in sample:
                draw_trajectories(
                    bgr, sample, horizon=horizon, scale=scale,
                    classify_cluster=classify_cluster,
                    color_override=_COLOR_GREEN_BGR if not classify_cluster else None,
                    traj_key="raw_traj_history", valid_key="raw_valid_history",
                    dashed=True,  # set False for solid history lines
                )
            draw_trajectories(
                bgr, sample, horizon=horizon, scale=scale,
                classify_cluster=classify_cluster,
            )
            draw_keypoints(bgr, sample, scale=scale,
                           classify_cluster=classify_cluster)
        draw_chunk_boundary(bgr, t, chunk_rec, prev_chunk_rec)
        draw_caption(bgr, chunk_rec, caption_field)
        draw_hud(bgr, t, n_frames, len(sample["keypoints"]) if sample else 0)

        writer.write(bgr)
        prev_chunk_rec = chunk_rec

    writer.release()
    print(f"Wrote {n_frames} frames to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def discover_episodes(root: Path) -> list[Path]:
    """Find episode directories under *root* that have unified format data."""
    out: list[Path] = []
    if (root / "images.npy").exists() and (root / "acceleration.npz").exists():
        return [root]
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "images.npy").exists():
            out.append(child)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate MP4 debug visualization for caption pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python visualize_debug.py outputs/droid/droid_shard00000_ep000
              python visualize_debug.py outputs/droid/droid_shard00000_ep000 --caption-field instruction_2
              python visualize_debug.py outputs/droid --all-videos --fps 10
        """),
    )
    p.add_argument(
        "path",
        type=Path,
        help="Episode directory (or dataset root with --all-videos).",
    )
    p.add_argument(
        "--all-videos",
        action="store_true",
        help="Treat PATH as a dataset root and process all episodes.",
    )
    p.add_argument(
        "--caption-field",
        default="instruction_1",
        help="Which caption field to overlay (default: instruction_1). "
             "Options: instruction_1, instruction_2, instruction_3, "
             "short_description, detailed_description, etc.",
    )
    p.add_argument(
        "--classify_cluster",
        action="store_true",
        help="Colour-code trajectories by cluster ID. Without this flag "
             "all future trajectories are drawn in red.",
    )
    p.add_argument(
        "--visualize_history",
        action="store_true",
        help="Also draw historical (past) trajectories in green.",
    )
    p.add_argument("--horizon", type=int, default=8,
                    help="Number of trajectory timesteps to draw (default: 8).")
    p.add_argument("--fps", type=int, default=10,
                    help="Output video framerate (default: 10).")
    p.add_argument("--scale", type=float, default=2.0,
                    help="Scale factor for output resolution (default: 2.0).")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write MP4s. Defaults to <episode>/curated/.",
    )
    return p.parse_args(argv)


_FIELD_SHORT = {
    "instruction_1": "inst1",
    "instruction_2": "inst2",
    "instruction_3": "inst3",
}


def _caption_field_suffix(field: str) -> str:
    return _FIELD_SHORT.get(field, field)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all_videos:
        episodes = discover_episodes(args.path)
        if not episodes:
            print(f"No episodes found under {args.path}", file=sys.stderr)
            return 1
    else:
        if not (args.path / "images.npy").exists():
            print(f"{args.path} does not contain images.npy", file=sys.stderr)
            return 1
        episodes = [args.path]

    suffix = _caption_field_suffix(args.caption_field)

    for ep in episodes:
        if args.all_videos:
            out_dir = args.output_dir or (args.path / "visualize_caption")
            out_path = out_dir / f"{ep.name}_{suffix}.mp4"
        else:
            out_dir = args.output_dir or (ep / "curated")
            out_path = out_dir / f"debug_{args.caption_field}.mp4"
        try:
            generate_video(
                ep,
                out_path,
                caption_field=args.caption_field,
                horizon=args.horizon,
                fps=args.fps,
                scale=args.scale,
                classify_cluster=args.classify_cluster,
                visualize_history=args.visualize_history,
            )
        except Exception as e:
            print(f"ERROR processing {ep}: {e}", file=sys.stderr)
            continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
