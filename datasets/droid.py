"""DROID RLDS/TFRecord dataset helpers for infer.py.

DROID stores each robot episode as one serialized ``tf.train.Example``
inside a TFRecord shard (``droid_101-train.tfrecord-XXXXX-of-02048``).
Per-step images live as a ``BytesList`` of JPEG-encoded byte strings
under keys like ``steps/observation/exterior_image_1_left``.

This module provides a tiny pure-Python TFRecord + Example reader so
infer.py can stream episodes without a TensorFlow dependency.
"""

import io
import json
import os
import struct
from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image


# ── Pure-Python protobuf primitives ───────────────────────────────────────

def _read_varint(buf, pos: int) -> Tuple[int, int]:
    shift = 0
    result = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _skip_field(buf, pos: int, wire: int) -> int:
    if wire == 0:
        _, pos = _read_varint(buf, pos)
    elif wire == 1:
        pos += 8
    elif wire == 2:
        length, pos = _read_varint(buf, pos)
        pos += length
    elif wire == 5:
        pos += 4
    else:
        raise ValueError(f"Unsupported protobuf wire type {wire}")
    return pos


def _parse_bytes_list(feat) -> List[bytes]:
    """Parse a ``Feature`` submessage and return the BytesList values.

    Feature { BytesList bytes_list = 1; ... }
    BytesList { repeated bytes value = 1; }
    """
    out: List[bytes] = []
    pos = 0
    while pos < len(feat):
        tag, pos = _read_varint(feat, pos)
        field, wire = tag >> 3, tag & 0x7
        if field == 1 and wire == 2:
            blen, pos = _read_varint(feat, pos)
            blist = feat[pos:pos + blen]
            pos += blen
            bpos = 0
            while bpos < len(blist):
                btag, bpos = _read_varint(blist, bpos)
                bfield, bwire = btag >> 3, btag & 0x7
                if bfield == 1 and bwire == 2:
                    vlen, bpos = _read_varint(blist, bpos)
                    out.append(bytes(blist[bpos:bpos + vlen]))
                    bpos += vlen
                else:
                    bpos = _skip_field(blist, bpos, bwire)
        else:
            pos = _skip_field(feat, pos, wire)
    return out


def _extract_bytes_feature(record: bytes, target_key: str) -> Optional[List[bytes]]:
    """Return the BytesList of the named feature from a ``tf.train.Example``.

    Walks the proto without materialising other features. Returns ``None``
    if the feature is not present.
    """
    buf = memoryview(record)
    target = target_key.encode("utf-8")
    pos = 0
    while pos < len(buf):
        tag, pos = _read_varint(buf, pos)
        field, wire = tag >> 3, tag & 0x7
        if field == 1 and wire == 2:  # Example.features
            flen, pos = _read_varint(buf, pos)
            features = buf[pos:pos + flen]
            pos += flen
            fpos = 0
            while fpos < len(features):
                ftag, fpos = _read_varint(features, fpos)
                ffield, fwire = ftag >> 3, ftag & 0x7
                if ffield == 1 and fwire == 2:  # map<string, Feature> entry
                    elen, fpos = _read_varint(features, fpos)
                    entry = features[fpos:fpos + elen]
                    fpos += elen
                    key_bytes = None
                    val_bytes = None
                    epos = 0
                    while epos < len(entry):
                        etag, epos = _read_varint(entry, epos)
                        efield, ewire = etag >> 3, etag & 0x7
                        if efield == 1 and ewire == 2:
                            klen, epos = _read_varint(entry, epos)
                            key_bytes = bytes(entry[epos:epos + klen])
                            epos += klen
                        elif efield == 2 and ewire == 2:
                            vlen, epos = _read_varint(entry, epos)
                            val_bytes = entry[epos:epos + vlen]
                            epos += vlen
                        else:
                            epos = _skip_field(entry, epos, ewire)
                    if key_bytes == target and val_bytes is not None:
                        return _parse_bytes_list(val_bytes)
                else:
                    fpos = _skip_field(features, fpos, fwire)
        else:
            pos = _skip_field(buf, pos, wire)
    return None


# ── TFRecord reader ───────────────────────────────────────────────────────
# Record format on disk:
#     uint64 length  uint32 length_crc  bytes[length] data  uint32 data_crc
# CRCs are skipped; corrupted shards will surface as proto-parse errors.

def _read_nth_record(fp, target_idx: int) -> Optional[bytes]:
    i = 0
    while True:
        hdr = fp.read(8)
        if len(hdr) < 8:
            return None
        length = struct.unpack("<Q", hdr)[0]
        fp.seek(4, 1)  # length CRC
        if i == target_idx:
            return fp.read(length)
        fp.seek(length + 4, 1)  # payload + data CRC
        i += 1


def _count_records(path: str) -> int:
    """Count records in a TFRecord file by walking lengths only."""
    n = 0
    with open(path, "rb") as fp:
        while True:
            hdr = fp.read(8)
            if len(hdr) < 8:
                return n
            length = struct.unpack("<Q", hdr)[0]
            fp.seek(length + 8, 1)  # length CRC + payload + data CRC
            n += 1


# ── Episode discovery ────────────────────────────────────────────────────

DROID_VERSION_DIR = "1.0.1"


def _resolve_split_dir(base_path: str) -> str:
    """Return the directory that contains the TFRecord shards.

    Accepts either the dataset root or its versioned subdirectory.
    """
    if os.path.isfile(os.path.join(base_path, "dataset_info.json")):
        return base_path
    candidate = os.path.join(base_path, DROID_VERSION_DIR)
    if os.path.isfile(os.path.join(candidate, "dataset_info.json")):
        return candidate
    raise FileNotFoundError(
        f"Could not locate DROID dataset_info.json under {base_path}. "
        f"Expected {base_path}/dataset_info.json or "
        f"{base_path}/{DROID_VERSION_DIR}/dataset_info.json"
    )


def find_droid_episodes(
    base_path: str,
    start_shard: int = 0,
    num_shards: Optional[int] = None,
) -> List[str]:
    """Return virtual paths ``<shard_path>::ep<idx>`` for each episode in
    the requested shard range.

    Trusts ``shardLengths`` from ``dataset_info.json`` rather than opening
    every TFRecord just to count records.
    """
    split_dir = _resolve_split_dir(base_path)
    with open(os.path.join(split_dir, "dataset_info.json")) as f:
        info = json.load(f)
    train_split = next(s for s in info["splits"] if s["name"] == "train")
    shard_lengths = [int(x) for x in train_split["shardLengths"]]
    n_total = len(shard_lengths)

    if start_shard >= n_total:
        return []
    end_shard = n_total if num_shards is None else min(start_shard + num_shards, n_total)

    template = train_split["filepathTemplate"]
    name = info["name"]
    fmt = info.get("fileFormat", "tfrecord")

    episodes: List[str] = []
    for shard_idx in range(start_shard, end_shard):
        shard_filename = (
            template
            .replace("{DATASET}", name)
            .replace("{SPLIT}", "train")
            .replace("{FILEFORMAT}", fmt)
            .replace("{SHARD_X_OF_Y}", f"{shard_idx:05d}-of-{n_total:05d}")
        )
        shard_path = os.path.join(split_dir, shard_filename)
        if not os.path.isfile(shard_path):
            logger.warning(f"DROID: missing shard {shard_path}, skipping")
            continue
        for ep_idx in range(shard_lengths[shard_idx]):
            episodes.append(f"{shard_path}::ep{ep_idx}")
    logger.info(
        f"DROID: discovered {len(episodes)} episodes across shards "
        f"[{start_shard}, {end_shard}) of {n_total} total"
    )
    return episodes


def parse_droid_video_path(virtual_path: str) -> Tuple[str, int]:
    """Split ``<shard_path>::ep<idx>`` into ``(shard_path, ep_idx)``."""
    if "::ep" not in virtual_path:
        raise ValueError(
            f"DROID: expected '<shard>::ep<idx>', got {virtual_path!r}"
        )
    shard_path, ep_str = virtual_path.rsplit("::ep", 1)
    return shard_path, int(ep_str)


def droid_video_name(virtual_path: str) -> str:
    """Deterministic, sortable output name for a DROID episode."""
    shard_path, ep_idx = parse_droid_video_path(virtual_path)
    base = os.path.basename(shard_path)
    # filename: droid_101-train.tfrecord-00000-of-02048
    parts = base.split("-")
    shard_token = parts[-3] if len(parts) >= 3 else "?????"
    return f"droid_shard{shard_token}_ep{ep_idx:03d}"


# ── Language / metadata ──────────────────────────────────────────────────

DROID_LANGUAGE_KEYS = (
    "steps/language_instruction",
    "steps/language_instruction_2",
    "steps/language_instruction_3",
)


def read_droid_meta(virtual_path: str) -> Tuple[str, dict]:
    """Return ``(description, meta_dict)`` for one DROID episode.

    ``description`` holds the episode's distinct non-empty language
    instructions, one per line (empty string if the episode has none).
    DROID repeats the instruction on every step, so the first non-empty
    step value of each key is used.
    """
    shard_path, ep_idx = parse_droid_video_path(virtual_path)
    with open(shard_path, "rb") as fp:
        record = _read_nth_record(fp, ep_idx)
    if record is None:
        raise IndexError(
            f"DROID: episode index {ep_idx} not found in {shard_path}"
        )

    instructions: List[str] = []
    for key in DROID_LANGUAGE_KEYS:
        values = _extract_bytes_feature(record, key) or []
        text = next(
            (v.decode("utf-8", errors="replace").strip() for v in values if v.strip()),
            "",
        )
        if text and text not in instructions:
            instructions.append(text)

    meta = {
        "dataset": "droid",
        "shard": os.path.basename(shard_path),
        "episode_index": ep_idx,
        "language_instructions": instructions,
    }
    return "\n".join(instructions), meta


# ── Frame loading ────────────────────────────────────────────────────────

def load_droid_frames(
    virtual_path: str,
    frame_step: int = 1,
    frame_range: Optional[Tuple[int, int]] = None,
    camera: str = "exterior_image_1_left",
    target_fps: float = 0.0,  # accepted for API parity; unused (RLDS has no fps)
) -> Tuple[torch.Tensor, None, List[str]]:
    """Load RGB frames for one DROID episode.

    Returns ``(video_tensor, mask, original_filenames)`` matching
    ``load_video_and_mask`` in infer.py. ``video_tensor`` is float32 in
    [0, 1] with shape (N, 3, H, W).
    """
    shard_path, ep_idx = parse_droid_video_path(virtual_path)
    feature_key = f"steps/observation/{camera}"

    with open(shard_path, "rb") as fp:
        record = _read_nth_record(fp, ep_idx)
    if record is None:
        raise IndexError(
            f"DROID: episode index {ep_idx} not found in {shard_path}"
        )

    jpeg_list = _extract_bytes_feature(record, feature_key)
    if not jpeg_list:
        raise ValueError(
            f"DROID: feature {feature_key!r} missing or empty in episode "
            f"{ep_idx} of {shard_path}"
        )

    total = len(jpeg_list)
    indices = list(range(total))
    if frame_range is not None:
        indices = indices[frame_range[0]:frame_range[1]]
    if frame_step > 1:
        indices = indices[::frame_step]

    frames = []
    filenames = []
    for idx in indices:
        img = Image.open(io.BytesIO(jpeg_list[idx])).convert("RGB")
        frames.append(torch.from_numpy(np.array(img)).float())
        filenames.append(f"frame_{idx:06d}")

    logger.info(
        f"DROID: loaded {len(frames)}/{total} frames from "
        f"{os.path.basename(shard_path)} ep{ep_idx} "
        f"(camera={camera}, frame_step={frame_step})"
    )

    video_tensor = torch.stack(frames)               # (N, H, W, 3)
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # (N, 3, H, W)
    video_tensor /= 255.0
    return video_tensor, None, filenames
