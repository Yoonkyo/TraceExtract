"""EgoVerse dataset helpers for infer.py.

Reads raw EgoVerse episodes stored as Zarr v3 sharded arrays
(zstd-compressed JPEG frames) and returns tensors compatible with
the ``load_video_and_mask`` interface used by infer.py.
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

import zstandard


# ── Episode discovery ──────────────────────────────────────────────────────

def find_egoverse_episodes(base_path: str) -> List[str]:
    """Return sorted list of episode directories under *base_path*.

    An episode directory is identified by the presence of a top-level
    ``zarr.json`` file.
    """
    base_path = os.path.abspath(base_path)
    episodes = []
    for name in sorted(os.listdir(base_path)):
        ep_dir = os.path.join(base_path, name)
        if os.path.isdir(ep_dir) and os.path.isfile(os.path.join(ep_dir, "zarr.json")):
            episodes.append(ep_dir)
    return episodes


# ── Zarr v3 shard reader ──────────────────────────────────────────────────

def _read_shard_index(fp, file_size: int, n_items: int):
    """Read the sharding_indexed index (at end of file).

    Returns list of (offset, nbytes) per inner chunk.
    """
    # index = n_items * (uint64 offset + uint64 nbytes) + 4 bytes crc32c
    index_size = n_items * 16 + 4
    fp.seek(file_size - index_size)
    index_data = fp.read(index_size)
    entries = []
    for i in range(n_items):
        offset, nbytes = struct.unpack_from("<QQ", index_data, i * 16)
        entries.append((offset, nbytes))
    return entries


def _decode_vlen_jpeg(compressed_bytes: bytes) -> np.ndarray:
    """Decompress one inner chunk (zstd) and decode the vlen-bytes JPEG."""
    dctx = zstandard.ZstdDecompressor()
    data = dctx.decompress(compressed_bytes, max_output_size=10 * 1024 * 1024)
    # vlen-bytes: uint32 count, then per item: uint32 length + raw bytes
    length = struct.unpack_from("<I", data, 4)[0]
    jpeg_data = data[8 : 8 + length]
    img = Image.open(io.BytesIO(jpeg_data))
    return np.array(img)


# ── Frame loading ─────────────────────────────────────────────────────────

def load_egoverse_frames(
    episode_path: str,
    target_fps: float = 10.0,
    frame_step: int = 1,
    frame_range: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, None, List[str]]:
    """Load RGB frames from an EgoVerse episode directory.

    Returns the same ``(video_tensor, mask, original_filenames)`` tuple as
    ``load_video_and_mask`` in infer.py so it can be used as a drop-in
    replacement.

    ``video_tensor`` is float32 in [0, 1] with shape (N, 3, H, W).
    """
    # Read episode metadata
    with open(os.path.join(episode_path, "zarr.json")) as f:
        meta = json.load(f)

    attrs = meta["attributes"]
    total_frames = attrs["total_frames"]
    source_fps = attrs.get("fps", 30)

    # Read image array metadata
    img_zarr_path = os.path.join(episode_path, "images.front_1", "zarr.json")
    with open(img_zarr_path) as f:
        img_meta = json.load(f)
    n_items = img_meta["shape"][0]
    shard_frames = img_meta["chunk_grid"]["configuration"]["chunk_shape"][0]
    sharding_cfg = next(
        c for c in img_meta["codecs"] if c["name"] == "sharding_indexed"
    )["configuration"]
    inner_frames = sharding_cfg["chunk_shape"][0]
    items_per_shard = shard_frames // inner_frames

    # Determine frame step from target_fps
    effective_step = frame_step
    if target_fps > 0 and source_fps > target_fps:
        effective_step = max(1, round(source_fps / target_fps))
        logger.info(
            f"EgoVerse FPS={source_fps}, target_fps={target_fps} "
            f"→ effective frame_step={effective_step}"
        )

    # Determine frame indices to load
    all_indices = list(range(min(total_frames, n_items)))
    if frame_range is not None:
        all_indices = all_indices[frame_range[0] : frame_range[1]]
    if effective_step > 1:
        all_indices = all_indices[::effective_step]

    # Read shard indices (may span multiple shards)
    shard_dir = os.path.join(episode_path, "images.front_1", "c")
    shard_cache: dict = {}

    def get_shard(shard_id: int):
        if shard_id not in shard_cache:
            path = os.path.join(shard_dir, str(shard_id))
            file_size = os.path.getsize(path)
            with open(path, "rb") as fp:
                index = _read_shard_index(fp, file_size, items_per_shard)
            shard_cache[shard_id] = (path, index)
        return shard_cache[shard_id]

    # Decode selected frames
    frames = []
    filenames = []
    SENTINEL = 0xFFFFFFFFFFFFFFFF
    for idx in all_indices:
        shard_id = idx // shard_frames
        inner_idx = (idx % shard_frames) // inner_frames
        path, index = get_shard(shard_id)
        offset, nbytes = index[inner_idx]
        if offset == SENTINEL or nbytes == SENTINEL:
            logger.warning(f"EgoVerse: missing chunk for frame {idx} in {path}")
            continue
        with open(path, "rb") as fp:
            fp.seek(offset)
            compressed = fp.read(nbytes)
        img_np = _decode_vlen_jpeg(compressed)
        frames.append(torch.from_numpy(img_np).float())
        filenames.append(f"frame_{idx:010d}")

    logger.info(
        f"EgoVerse: loaded {len(frames)}/{total_frames} frames "
        f"(step={effective_step}) from {os.path.basename(episode_path)}"
    )

    video_tensor = torch.stack(frames)  # (N, H, W, 3)
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # (N, 3, H, W)
    video_tensor /= 255.0

    return video_tensor, None, filenames
