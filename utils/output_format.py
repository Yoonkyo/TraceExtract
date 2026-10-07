"""Helpers for loading TraceExtract output data in both the legacy per-frame
format (``images/*.png``, ``samples/*.npz``) and the unified ``.npy`` format
(``images.npy``, ``depth.npy``, ``samples/*.npy``).

All public functions auto-detect which format is present on disk and load
accordingly, so consumer scripts work with both old and new outputs.
"""

import os
import numpy as np
from PIL import Image


# ---- Format detection -------------------------------------------------------

def is_unified_format(video_dir):
    """Return True if *video_dir* uses the unified .npy output format."""
    return os.path.isfile(os.path.join(video_dir, "images.npy"))


def has_unified_samples(video_dir):
    """Return True if *video_dir* has unified sample .npy files."""
    return os.path.isfile(os.path.join(video_dir, "samples", "frame_indices.npy"))


# ---- Image loading -----------------------------------------------------------

def load_image(video_dir, video_name, frame_idx):
    """Load a single RGB frame as ``(H, W, 3)`` uint8 ndarray."""
    images_npy = os.path.join(video_dir, "images.npy")
    if os.path.isfile(images_npy):
        images = np.load(images_npy, mmap_mode="r")
        return np.array(images[frame_idx])  # copy out of mmap
    # Legacy per-frame PNG
    png_path = os.path.join(video_dir, "images", f"{video_name}_{frame_idx}.png")
    return np.array(Image.open(png_path))


def get_image_shape(video_dir, video_name):
    """Return ``(T, H, W)`` without loading full pixel data."""
    images_npy = os.path.join(video_dir, "images.npy")
    if os.path.isfile(images_npy):
        images = np.load(images_npy, mmap_mode="r")
        T, H, W = images.shape[:3]
        return T, H, W
    # Legacy: peek at a single PNG
    img_dir = os.path.join(video_dir, "images")
    for fname in sorted(os.listdir(img_dir)):
        if fname.endswith(".png"):
            with Image.open(os.path.join(img_dir, fname)) as im:
                W, H = im.size
            T = len([f for f in os.listdir(img_dir) if f.endswith(".png")])
            return T, H, W
    raise FileNotFoundError(f"No image data found in {video_dir}")


# ---- Depth loading -----------------------------------------------------------

def load_depth(video_dir, video_name, frame_idx):
    """Load a single depth frame as ``(H, W)`` float32 (meters)."""
    depth_npy = os.path.join(video_dir, "depth.npy")
    if os.path.isfile(depth_npy):
        depths = np.load(depth_npy, mmap_mode="r")
        return np.array(depths[frame_idx]).astype(np.float32)
    # Legacy per-frame PNG / NPZ
    base = os.path.join(video_dir, "depth", f"{video_name}_{frame_idx}")
    raw_npz = f"{base}_raw.npz"
    if os.path.isfile(raw_npz):
        with np.load(raw_npz) as d:
            return d["depth"]
    png_path = f"{base}.png"
    if os.path.isfile(png_path):
        depth_img = np.array(Image.open(png_path))
        return depth_img.astype(np.float32) / 10000.0
    raise FileNotFoundError(f"No depth data for frame {frame_idx} in {video_dir}")


# ---- Sample loading ----------------------------------------------------------

def load_sample(video_dir, video_name, frame_idx):
    """Load the sample data for *frame_idx* as a dict of numpy arrays.

    Returns a dict with the same keys as the legacy ``.npz`` files:
    ``keypoints``, ``traj``, ``traj_history``, ``valid_steps``, etc.

    Raises ``KeyError`` if *frame_idx* is not present in the data.
    """
    samples_dir = os.path.join(video_dir, "samples")

    # ---- Unified .npy format ------------------------------------------------
    fi_path = os.path.join(samples_dir, "frame_indices.npy")
    if os.path.isfile(fi_path):
        frame_indices = np.load(fi_path)
        offsets = np.load(os.path.join(samples_dir, "offsets.npy"))

        matches = np.where(frame_indices == frame_idx)[0]
        if len(matches) == 0:
            raise KeyError(
                f"Frame {frame_idx} not found in unified samples "
                f"(available: {frame_indices.tolist()[:10]}...)"
            )
        slot = int(matches[0])
        lo, hi = int(offsets[slot]), int(offsets[slot + 1])

        # Load each array with mmap and slice
        def _load_slice(name):
            arr = np.load(os.path.join(samples_dir, f"{name}.npy"), mmap_mode="r")
            return np.array(arr[lo:hi])  # copy out of mmap

        return {
            "frame_index": np.array([frame_idx]),
            "keypoints": _load_slice("keypoints"),
            "cluster_ids": _load_slice("cluster_ids"),
            "is_moving": _load_slice("is_moving"),
            "visibs": _load_slice("visibs"),
            "traj": _load_slice("traj"),
            "traj_history": _load_slice("traj_history"),
            "valid_steps": _load_slice("valid_steps"),
            "valid_steps_history": _load_slice("valid_steps_history"),
            "raw_traj": _load_slice("raw_traj"),
            "raw_traj_history": _load_slice("raw_traj_history"),
            "raw_valid_steps": _load_slice("raw_valid_steps"),
            "raw_valid_steps_history": _load_slice("raw_valid_steps_history"),
        }

    # ---- Legacy per-frame .npz format ---------------------------------------
    npz_path = os.path.join(samples_dir, f"{video_name}_{frame_idx}.npz")
    if os.path.isfile(npz_path):
        data = np.load(npz_path)
        result = {k: data[k] for k in data.files}
        data.close()
        return result

    raise FileNotFoundError(
        f"No sample data for frame {frame_idx} in {video_dir}"
    )


def list_sample_frame_indices(video_dir, video_name):
    """Return a sorted list of frame indices that have sample data."""
    samples_dir = os.path.join(video_dir, "samples")

    fi_path = os.path.join(samples_dir, "frame_indices.npy")
    if os.path.isfile(fi_path):
        return sorted(np.load(fi_path).tolist())

    # Legacy: parse filenames
    indices = []
    if os.path.isdir(samples_dir):
        for fname in os.listdir(samples_dir):
            if fname.endswith(".npz") and fname.startswith(f"{video_name}_"):
                try:
                    idx = int(fname[:-4].split("_")[-1])
                    indices.append(idx)
                except ValueError:
                    pass
    return sorted(indices)
