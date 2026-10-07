"""Registry of dataset-specific loaders used by infer.py (``--dataset_name``).

Each dataset provides the same four hooks, so infer.py handles every dataset
through one code path:

* ``find_episodes(args)``  → list of episode paths under ``args.video_path``
* ``episode_name(path)``   → unique output directory name for an episode
* ``load_frames(path, args)`` → ``(video_tensor, mask, filenames)``, or
  ``None`` to use infer.py's generic video / frame-folder loader
* ``read_meta(path)``      → ``(description, meta_dict)`` written to
  ``description.txt`` / ``meta.json``, or ``None`` if the dataset has no
  per-episode annotation

Imports are deferred so that a dataset's extra dependencies (``h5py`` for
EgoDex, ``zstandard`` for EgoVerse) are only needed when that dataset is used.

To add a new dataset, write ``datasets/<name>.py`` with the hooks above and
register a ``DatasetSpec`` in ``DATASETS``.
"""

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    find_episodes: Callable
    episode_name: Callable
    load_frames: Optional[Callable] = None
    read_meta: Optional[Callable] = None


# ── AgiBot ──────────────────────────────────────────────────────────────

def _agibot_find(args):
    from datasets.agibot import find_agibot_episodes
    return find_agibot_episodes(args.video_path)


def _agibot_name(path):
    from datasets.agibot import agibot_video_name
    return agibot_video_name(path)


def _agibot_meta(path):
    from datasets.agibot import read_agibot_meta
    return read_agibot_meta(path)


# ── DROID ───────────────────────────────────────────────────────────────

def _droid_find(args):
    from datasets.droid import find_droid_episodes
    return find_droid_episodes(
        args.video_path,
        start_shard=args.droid_start_shard,
        num_shards=args.droid_num_shards,
    )


def _droid_name(path):
    from datasets.droid import droid_video_name
    return droid_video_name(path)


def _droid_load(path, args):
    from datasets.droid import load_droid_frames
    return load_droid_frames(path, frame_step=args.frame_step, camera=args.droid_camera)


def _droid_meta(path):
    from datasets.droid import read_droid_meta
    return read_droid_meta(path)


# ── EgoDex ──────────────────────────────────────────────────────────────

def _egodex_find(args):
    from datasets.egodex import find_egodex_episodes
    return find_egodex_episodes(args.video_path)


def _egodex_name(path):
    from datasets.egodex import egodex_video_name
    return egodex_video_name(path)


def _egodex_meta(path):
    from datasets.egodex import read_egodex_meta
    return read_egodex_meta(path)


# ── EgoVerse ────────────────────────────────────────────────────────────

def _egoverse_find(args):
    from datasets.egoverse import find_egoverse_episodes
    return find_egoverse_episodes(args.video_path)


def _egoverse_load(path, args):
    from datasets.egoverse import load_egoverse_frames
    return load_egoverse_frames(path, target_fps=args.target_fps, frame_step=args.frame_step)


DATASETS: Dict[str, DatasetSpec] = {
    "agibot": DatasetSpec("agibot", _agibot_find, _agibot_name, read_meta=_agibot_meta),
    "droid": DatasetSpec("droid", _droid_find, _droid_name, _droid_load, _droid_meta),
    "egodex": DatasetSpec("egodex", _egodex_find, _egodex_name, read_meta=_egodex_meta),
    "egoverse": DatasetSpec(
        "egoverse", _egoverse_find, lambda p: os.path.basename(p.rstrip("/")), _egoverse_load
    ),
}


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name]
