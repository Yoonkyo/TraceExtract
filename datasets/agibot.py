"""AgiBot dataset helpers for infer.py.

AgiBot (chunked) layout: ``<root>/<task_id>/<episode_id>/{obs.mp4, task.txt}``.
Frames are loaded via the default ``load_video_and_mask`` path in infer.py
(standard mp4); this module only handles episode discovery, output naming,
and the per-episode task text.
"""

import glob
import os
from typing import Any, Dict, List, Tuple


def find_agibot_episodes(root: str) -> List[str]:
    """Return sorted ``obs.mp4`` paths under ``<root>/*/*/obs.mp4``."""
    root = os.path.abspath(root)
    return sorted(glob.glob(os.path.join(root, "*", "*", "obs.mp4")))


def agibot_video_name(mp4_path: str) -> str:
    """``.../<task_id>/<episode_id>/obs.mp4`` → ``<episode_id>``."""
    return os.path.basename(os.path.dirname(mp4_path))


def read_agibot_meta(mp4_path: str) -> Tuple[str, Dict[str, Any]]:
    """Return ``(task_text, meta_dict)`` from the sibling ``task.txt``."""
    ep_dir = os.path.dirname(mp4_path)
    task_path = os.path.join(ep_dir, "task.txt")
    primary = ""
    if os.path.isfile(task_path):
        with open(task_path, encoding="utf-8") as f:
            primary = f.read().strip()
    meta = {
        "dataset": "agibot",
        "task_id": os.path.basename(os.path.dirname(ep_dir)),
        "episode_id": os.path.basename(ep_dir),
    }
    return primary, meta
