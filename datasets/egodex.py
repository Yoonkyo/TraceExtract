"""EgoDex dataset helpers for infer.py.

EgoDex layout: ``<root>/{part1..part4}/<task_dir>/<id>.mp4`` with a paired
``<id>.hdf5`` sidecar holding the annotation. Frames are loaded via the
default ``load_video_and_mask`` path in infer.py (standard mp4); this module
only handles episode discovery, output naming, and annotation extraction.
"""

import glob
import os
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np


# EgoDex videos are recorded at a fixed 30 fps on Vision Pro.
SOURCE_FPS = 30


def find_egodex_episodes(root: str) -> List[str]:
    """Return sorted list of mp4 paths under ``<root>/part*/<task_dir>/*.mp4``.

    Episodes without a paired ``.hdf5`` sidecar are dropped.
    """
    root = os.path.abspath(root)
    pattern = os.path.join(root, "*", "*", "*.mp4")
    mp4s = sorted(glob.glob(pattern))
    return [p for p in mp4s if os.path.isfile(p[:-4] + ".hdf5")]


def egodex_video_name(mp4_path: str) -> str:
    """Build a globally-unique name from an EgoDex mp4 path.

    ``.../part1/add_remove_lid/0.mp4`` → ``part1__add_remove_lid__0``.
    Episode IDs repeat across parts, so all three components are required.
    """
    episode_id = os.path.splitext(os.path.basename(mp4_path))[0]
    task_dir = os.path.basename(os.path.dirname(mp4_path))
    part = os.path.basename(os.path.dirname(os.path.dirname(mp4_path)))
    return f"{part}__{task_dir}__{episode_id}"


def _to_py(val: Any) -> Any:
    """Convert h5py attribute values to JSON-serializable Python types."""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray):
        return [_to_py(x) for x in val.tolist()]
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def read_egodex_meta(mp4_path: str) -> Tuple[str, Dict[str, Any]]:
    """Read the hdf5 sidecar and return ``(primary_description, meta_dict)``.

    ``primary_description`` is the string written to ``description.txt``
    (the value of ``llm_description``, falling back to ``task`` if missing).
    ``meta_dict`` is the full structured annotation written to ``meta.json``.
    """
    hdf5_path = mp4_path[:-4] + ".hdf5"
    episode_id = os.path.splitext(os.path.basename(mp4_path))[0]
    task_dir = os.path.basename(os.path.dirname(mp4_path))
    part = os.path.basename(os.path.dirname(os.path.dirname(mp4_path)))

    meta: Dict[str, Any] = {
        "dataset": "egodex",
        "part": part,
        "task_dir": task_dir,
        "episode_id": episode_id,
        "source_fps": SOURCE_FPS,
    }

    with h5py.File(hdf5_path, "r") as f:
        for k, v in f.attrs.items():
            meta[k] = _to_py(v)
        if "transforms/camera" in f:
            meta["num_frames_at_source_fps"] = int(f["transforms/camera"].shape[0])

    primary = meta.get("llm_description") or meta.get("task") or ""
    return str(primary), meta
