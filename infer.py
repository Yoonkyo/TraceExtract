import os
import shutil
import socket
import math
import time
import numpy as np
import cv2
import mediapy as media
import torch
from PIL import Image
import tqdm
import glob
import random
import argparse
from loguru import logger
import json
import re
import colorsys

from utils.video_depth_pose_utils import video_depth_pose_dict
from datasets.registry import DATASETS, get_dataset

from datasets.data_ops import _filter_one_depth
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
from utils.inference_utils import load_model, inference
from utils.threed_utils import (
    project_tracks_3d_to_2d,
    project_tracks_3d_to_3d,
)
from utils.dino_keypoints import create_dino_extractor, extract_dino_keypoints


def _move_model_to(model, device):
    """Move a model to a device, handling special cases."""
    if hasattr(model, 'to_device'):
        # VGGT4Wrapper has its own to_device method
        model.to_device(device)
    else:
        model.to(device)
        # Update self.device if the model tracks it (e.g. DINO extractor)
        if hasattr(model, 'device'):
            model.device = torch.device(device) if isinstance(device, str) else device
    if str(device) == "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _log_vram(tag: str):
    """Log current GPU VRAM usage with a descriptive tag."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"[VRAM] {tag}: {alloc:.2f} GiB allocated, {reserved:.2f} GiB reserved, {total:.2f} GiB total")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to video directory (for batch processing) or single video folder",
    )
    parser.add_argument(
        "--depth_path",
        type=str,
        default=None,
        help="Path to depth directory (if known depth is provided) for batch processing or single video folder",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoints", "tapip3d_final.pth"),
        help="TAPIP3D checkpoint (default: checkpoints/tapip3d_final.pth in this repo)",
    )
    parser.add_argument('--depth_pose_method', type=str, default='vggt4', choices=video_depth_pose_dict.keys(),
                        help="Depth + camera-pose estimator (VGGT4Track from SpatialTrackerV2).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_iters", type=int, default=6,
                        help="TAPIP3D refinement iterations per tracking call.")
    parser.add_argument("--out_dir", type=str, default="outputs",
                        help="Output root; one sub-directory is written per episode.")
    parser.add_argument("--chunk_size", type=int, default=60,
                        help="Frames per VGGT dense chunk. Controls VGGT memory usage.")
    parser.add_argument("--tracking_chunk_size", type=int, default=None,
                        help="Frames per tracking (TAPIP3D) chunk. Defaults to chunk_size if not set. "
                             "Tracking uses less VRAM, so this can be larger than chunk_size.")
    parser.add_argument("--sparse_max", type=int, default=150,
                        help="Max frames for the global sparse VGGT pass "
                             "(anchor frames for coordinate alignment).")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Take every Nth frame (1 = no skip). Used for frame folders and "
                             "DROID; for video files and EgoVerse only when --target_fps is 0.")
    parser.add_argument("--target_fps", type=float, default=10.0,
                        help="Target FPS for video inputs. If the video's FPS is higher, "
                             "frames are subsampled to approximate this rate. "
                             "If the video's FPS is already <= target, no subsampling is done. "
                             "Set to 0 to disable and use --frame_step instead.")
    parser.add_argument("--target_hw", type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="Resize loaded frames to (H, W) before VGGT/tracking. "
                             "Saved images.npy/depth.npy/cameras.npz inherit this "
                             "resolution. Use e.g. --target_hw 360 640 for egodex "
                             "to cut disk usage ~9x.")
    parser.add_argument("--min_track_coverage", type=float, default=0.6,
                        help="Minimum average track coverage (0-1) to accept a video. "
                             "Videos below this threshold are skipped as low quality.")
    parser.add_argument("--save_video", action="store_true", default=False,
                        help="With --save_main_npz, also store the RGB video in the legacy .npz.")
    parser.add_argument(
        "--save_main_npz", action="store_true", default=False,
        help="Also save the legacy per-video <name>.npz bundle (coords/depths/etc.). "
             "Off by default since downstream tools now read cameras.npz + samples/.",
    )
    parser.add_argument(
        "--batch_process",
        action="store_true",
        default=False,
        help="Process all video folders in the given directory",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip processing if output already exists",
    )
    parser.add_argument(
        "--scan_depth",
        type=int,
        default=2,  # default depth changed to 2
        help="How many directory levels below --video_path to scan for subfolders "
            "when --batch_process is enabled. Default is 2 (e.g., P02_02_01)."
    )
    parser.add_argument(
        "--future_len",
        type=int,
        default=128,
        help="Future horizon (frames) saved per query frame",
    )
    parser.add_argument(
        "--history_len",
        type=int,
        default=32,
        help="History horizon (frames) saved per query frame",
    )
    # ── DINO keypoint extraction ──
    parser.add_argument("--clustering_method", type=str, default='bipartite',
                        choices=['bipartite', 'kmeans'],
                        help="Clustering method for DINO features")
    parser.add_argument("--n_clusters", type=int, default=16,
                        help="Number of clusters for kmeans")
    parser.add_argument("--num_points_per_entity", type=int, default=48,
                        help="Keypoints to sample per cluster")
    parser.add_argument("--merge_ratio", type=int, default=25,
                        help="Merge ratio for bipartite clustering")
    parser.add_argument("--clustering_num_iters", type=int, default=11,
                        help="Iterations for bipartite clustering")
    parser.add_argument("--use_connected_components", action="store_true",
                        help="Use connected components in DINO clustering")
    parser.add_argument("--dino_stride", type=int, default=5,
                        help="Frame stride for DINO feature extraction (default: 5)")
    # ── Dataset ──
    parser.add_argument("--dataset_name", type=str, default=None,
                        choices=sorted(DATASETS),
                        help="Dataset-specific loader (see datasets/registry.py). If set, "
                             "--video_path is the raw dataset root and every episode "
                             "found under it is processed.")
    parser.add_argument("--episode_list", type=str, default=None,
                        help="Optional text file of episode names (one per line; the "
                             "output directory names). Only listed episodes are processed.")
    # ── DROID-specific options ──
    parser.add_argument("--droid_start_shard", type=int, default=0,
                        help="DROID: index of first TFRecord shard to process.")
    parser.add_argument("--droid_num_shards", type=int, default=None,
                        help="DROID: number of consecutive shards to process. "
                             "Default None = all remaining shards.")
    parser.add_argument("--droid_camera", type=str, default="exterior_image_1_left",
                        choices=["exterior_image_1_left", "exterior_image_2_left",
                                 "wrist_image_left"],
                        help="DROID: which observation camera to feed the pipeline.")
    # ── Debug ──
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Save debug MP4 with tracked points overlaid")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Truncate the discovered episode list to the first "
                             "N entries (after shuffling). Intended for dry-runs.")
    return parser.parse_args()

def retarget_trajectories(
    trajectory: np.ndarray,
    interval: float = 0.05,
    max_length: int = 64,
    top_percent: float = 0.02,
    image_width: float = None,
    image_height: float = None,
):
    """
    Synchronous arc-length retargeting using per-segment robust speeds.

    Steps:
      1) Global normalize x,y by (trajectory[-1,0,0], trajectory[-1,0,1]); no clipping (off-screen motion preserved).
      2) For each time segment t: compute lengths for all tracks; take mean of top `top_percent`
         → robust_seglen[t].
      3) Build cumulative arc-length from robust_seglen and place targets every `interval`.
         (Long segments get subdivided; short ones merge implicitly.)
      4) For each target in segment t with fraction alpha, interpolate *all* tracks
         between frames t and t+1 with the same alpha (synchronous).
      5) Denormalize x,y only; z (if present) is linearly interpolated without scaling.

    Args:
        trajectory: (N, H, D) with D in {2,3}
        interval: target arc-length step
        max_length: output max length
        top_percent: fraction (0,1] for robust top-k mean per segment (e.g., 0.02 = top 2%)
        image_width: if given, normalize x by this value instead of trajectory[-1,0,0]
        image_height: if given, normalize y by this value instead of trajectory[-1,0,1]

    Returns:
        retargeted: (N, max_length, D), padded with -np.inf
        valid_mask: (max_length) bool
    """
    assert trajectory.ndim == 3, "trajectory must be (N, H, D)"
    N, H, D = trajectory.shape
    assert D in (2, 3), "D must be 2 or 3"
    if not (0 < top_percent <= 1.0):
        raise ValueError("top_percent must be in (0, 1].")
    if interval <= 0:
        raise ValueError("interval must be > 0")
    if H < 2:
        # If H==1, there is no segment to interpolate → return only the first frame
        ret = np.full((N, max_length, D), -np.inf, dtype=trajectory.dtype)
        mask = np.zeros((max_length), dtype=bool)
        ret[:, 0, :] = trajectory[:, 0, :]
        mask[0] = True
        return ret, mask

    eps = 1e-12

    # ---- 1) Global normalization (x,y) & clipping ----
    if image_width is not None and image_height is not None:
        scale_x = float(image_width)
        scale_y = float(image_height)
    else:
        scale_x = float(trajectory[-1, 0, 0])
        scale_y = float(trajectory[-1, 0, 1])
    if abs(scale_x) < eps: scale_x = 1.0
    if abs(scale_y) < eps: scale_y = 1.0

    traj_norm = trajectory.astype(np.float64, copy=True)
    traj_norm[:, :, 0] /= scale_x
    traj_norm[:, :, 1] /= scale_y
    # No clipping: off-screen motion from 3D tracking is real movement
    # and should be preserved in the retargeted output.

    # ---- 2) Robust length per segment t: mean of top k% ----
    # seglens_all: (N, H-1)
    diffs_all = traj_norm[:, 1:, :] - traj_norm[:, :-1, :]
    seglens_all = np.linalg.norm(diffs_all, axis=2)

    k = max(1, int(np.ceil(top_percent * N)))
    # Use np.partition to get per-segment (column-wise) top-k without full sorting
    # Values below index N-k are smaller; values at/above are larger
    part = np.partition(seglens_all, N - k, axis=0)      # (N, H-1)
    topk = part[N - k:, :]                                # (k, H-1)
    robust_seglen = topk.mean(axis=0)                     # (H-1,)

    total_len = float(robust_seglen.sum())
    # Output buffers
    retargeted = np.full((N, max_length, D), -np.inf, dtype=trajectory.dtype)
    valid_mask = np.zeros((max_length), dtype=bool)

    # ---- 3) Create targets at 'interval' along the robust cumulative length ----
    k_max = int(np.floor(total_len / interval))
    num_samples = min(k_max + 1, max_length)
    targets = interval * np.arange(num_samples, dtype=np.float64)
    targets[-1] = min(targets[-1], total_len)

    # Cumulative length s (vertex-based): s[0]=0, s[i]=sum_{j<i} robust_seglen[j]
    s = np.zeros((H,), dtype=np.float64)
    s[1:] = np.cumsum(robust_seglen, dtype=np.float64)

    # Segment index and in-segment fraction alpha for each target
    idx_seq = np.searchsorted(s, targets, side='right') - 1   # (num_samples,)
    idx_seq = np.clip(idx_seq, 0, H - 2)
    denom = np.maximum(robust_seglen[idx_seq], eps)           # (num_samples,)
    alpha = (targets - s[idx_seq]) / denom                    # (num_samples,)
    alpha_seq = alpha.reshape(-1, 1)                          # (num_samples,1)

    # ---- 4) Synchronous interpolation: apply the same (idx, alpha) to all tracks ----
    left = traj_norm[:, idx_seq, :]           # (N, num_samples, D)
    right = traj_norm[:, idx_seq + 1, :]      # (N, num_samples, D)
    samples_norm = left + alpha_seq[None, :, :] * (right - left)  # (N, num_samples, D)

    # ---- 5) Denormalize: scale only x,y back ----
    samples_out = samples_norm.astype(trajectory.dtype, copy=True)
    samples_out[:, :, 0] *= scale_x
    samples_out[:, :, 1] *= scale_y
    # Keep z as the linear interpolation result

    L = num_samples
    retargeted[:, :L, :] = samples_out
    valid_mask[:L] = True
    return retargeted, valid_mask


# ---------------------------------------------------------------------------
#  Track coverage quality metric
# ---------------------------------------------------------------------------

def compute_track_coverage(query_frame_results, full_intrinsics, full_extrinsics,
                           height, width, patch_grid=16):
    """Compute per-frame track coverage on a patch grid.

    For each frame, projects all visible tracked keypoints onto a (patch_grid x
    patch_grid) grid and returns the fraction of patches that have at least one
    visible track.

    Returns:
        per_frame_coverage: (T,) float array, coverage ratio per frame.
        mean_coverage: float, average across all frames.
    """
    intrs_np = full_intrinsics.cpu().numpy() if torch.is_tensor(full_intrinsics) else full_intrinsics
    extrs_np = full_extrinsics.cpu().numpy() if torch.is_tensor(full_extrinsics) else full_extrinsics
    T = intrs_np.shape[0]

    covered = np.zeros((T, patch_grid, patch_grid), dtype=bool)

    for _pf, data in query_frame_results.items():
        coords = data['coords']
        visibs_d = data['visibs']
        frame_indices = data['frame_indices']

        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
        visibs_np = visibs_d.cpu().numpy() if torch.is_tensor(visibs_d) else visibs_d

        for t_local, t_global in enumerate(frame_indices):
            vis = visibs_np[t_local] > 0.5
            if not np.any(vis):
                continue
            coords_3d = coords_np[t_local]  # (N, 3)
            camera_view = {
                'K': intrs_np[t_global],
                'c2w': np.linalg.inv(extrs_np[t_global]),
                'height': height, 'width': width,
            }
            proj_2d = project_tracks_3d_to_2d(
                tracks3d=coords_3d[np.newaxis],
                camera_views=[camera_view],
            )[0]  # (N, 2)
            px = proj_2d[vis, 0]
            py = proj_2d[vis, 1]
            # Filter to in-bounds points
            in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            px, py = px[in_bounds], py[in_bounds]
            patch_x = np.clip((px / width * patch_grid).astype(int), 0, patch_grid - 1)
            patch_y = np.clip((py / height * patch_grid).astype(int), 0, patch_grid - 1)
            covered[t_global, patch_y, patch_x] = True

    per_frame_coverage = covered.reshape(T, -1).mean(axis=1)
    mean_coverage = per_frame_coverage.mean()
    return per_frame_coverage, mean_coverage


# ---------------------------------------------------------------------------
#  Motion detection
# ---------------------------------------------------------------------------

def compute_is_moving(coords_3d, visibs, intrinsics, extrinsics,
                      height, width, threshold=40.0, depth_weight=0.1,
                      horizon=64):
    """Decide which keypoints are moving based on maximum pairwise
    displacement in projected (u, v, z) space through a fixed camera.

    Projects all 3-D world coordinates through the first frame's camera
    to obtain (u_pixel, v_pixel, z_depth), then for each keypoint finds
    the maximum Euclidean distance between any two visible frames.
    This captures the full extent of motion (e.g. a point that swings
    out and returns) while being robust to accumulated tracking jitter
    on stationary surfaces.

    Args:
        coords_3d: (T, N, 3) tracked 3-D world coordinates.
        visibs: (T, N) visibility flags.
        intrinsics: (T, 3, 3) camera intrinsics.
        extrinsics: (T, 4, 4) camera extrinsics (world-to-camera).
        height, width: image dimensions.
        threshold: max pairwise displacement in (u, v, z) space above
            which a point is considered ``moving`` (default: 40.0 pixels).
        depth_weight: weight applied to the z (depth) component before
            computing displacement norms (default: 0.05).  Lower values
            make the filter rely more on u/v pixel motion.
        horizon: only use the first ``horizon`` frames to compute
            displacement (default: 64).  Set to 0 or None to use all.

    Returns:
        is_moving: (N,) boolean ndarray.
        max_disp: (N,) float ndarray – raw max pairwise displacement in
            (u, v, weighted-z) pixel space for each keypoint.
    """
    coords_np = coords_3d.cpu().numpy() if torch.is_tensor(coords_3d) else coords_3d
    visibs_np = (visibs.cpu().numpy() if torch.is_tensor(visibs) else visibs).astype(bool)
    intrs_np = intrinsics.cpu().numpy() if torch.is_tensor(intrinsics) else intrinsics
    extrs_np = extrinsics.cpu().numpy() if torch.is_tensor(extrinsics) else extrinsics

    T = coords_np.shape[0]
    if horizon:
        T = min(T, horizon)

    # Project through first frame's camera (fixed viewpoint → camera
    # motion is factored out).
    camera_view_0 = {
        'K': intrs_np[0],
        'c2w': np.linalg.inv(extrs_np[0]),
        'height': height,
        'width': width,
    }
    tracks_uvz = project_tracks_3d_to_3d(
        tracks3d=coords_np[:T],
        camera_views=[camera_view_0] * T,
    )                                                       # (T, N, 3)

    # Down-weight depth component so u/v pixel motion dominates
    tracks_uvz[:, :, 2] *= depth_weight

    # Max pairwise displacement per keypoint across visible frames.
    # For each keypoint, find the maximum distance between any two
    # frames where it is visible — this is the trajectory "diameter".
    N = tracks_uvz.shape[1]
    max_disp = np.zeros(N)
    for n in range(N):
        valid = visibs_np[:T, n]
        if valid.sum() < 2:
            continue
        pts = tracks_uvz[valid, n, :]                       # (V, 3)
        diffs = pts[:, None, :] - pts[None, :, :]           # (V, V, 3)
        dists = np.linalg.norm(diffs, axis=2)               # (V, V)
        max_disp[n] = dists.max()

    return max_disp > threshold, max_disp                     # (N,), (N,)


def compute_per_frame_acceleration(
    query_frame_results,
    is_moving_per_frame=None,
    T_total=None,
    margin=5,
    depth_weight=0.1,
):
    """Compute per-frame average acceleration magnitude in projected (u, v, z) space.

    For each moving keypoint, 3-D world coordinates are projected through
    the segment's first frame camera to obtain (u, v, z) — pixel position
    plus depth.  Velocity at frame *t* is ``uvz[t+1] - uvz[t]``.
    Acceleration at frame *t* is ``||vel[t+1] - vel[t]||``, capturing
    both speed changes and direction changes.

    Working in projected space makes the metric robust to noisy 3-D
    reconstruction of background points (their pixel positions are stable)
    while still factoring out camera motion (fixed viewpoint).

    Only keypoints with at least ``margin`` consecutive visible frames
    on both sides of each timestep are included, to avoid spurious
    spikes from tracks that just appeared or are about to disappear.

    Args:
        query_frame_results: dict of segment data (same as used elsewhere).
        is_moving_per_frame: dict  peak_frame -> (N,) bool.
        T_total: total number of frames in the video.
        margin: require this many consecutive visible frames before *and*
            after each timestep for a keypoint to contribute.
        depth_weight: weight applied to the z (depth) component before
            computing velocity/acceleration norms (default: 0.2).

    Returns:
        accel_per_frame: ``(T_total,)`` float32 array.  Frames with no
            valid data are 0.  Values are in weighted (u, v, z) units per
            frame squared.
    """
    # Pre-extract per-segment numpy arrays
    segments = []
    for peak_frame, fdata in query_frame_results.items():
        coords_np = (fdata['coords'].cpu().numpy()
                     if torch.is_tensor(fdata['coords']) else fdata['coords'])
        visibs_np = (fdata['visibs'].cpu().numpy()
                     if torch.is_tensor(fdata['visibs']) else fdata['visibs'])
        intrs_np = (fdata['intrinsics_segment'].cpu().numpy()
                    if torch.is_tensor(fdata['intrinsics_segment'])
                    else fdata['intrinsics_segment'])
        extrs_np = (fdata['extrinsics_segment'].cpu().numpy()
                    if torch.is_tensor(fdata['extrinsics_segment'])
                    else fdata['extrinsics_segment'])
        fi = fdata.get('frame_indices',
                       list(range(coords_np.shape[0])))
        is_mov = ((is_moving_per_frame or {}).get(
            peak_frame, np.ones(coords_np.shape[1], dtype=bool)))
        segments.append({
            'coords': coords_np,   # (T_seg, N, 3) world frame
            'visibs': visibs_np.astype(bool),
            'intrs': intrs_np,
            'extrs': extrs_np,
            'frame_indices': fi,
            'is_moving': np.asarray(is_mov, dtype=bool),
        })

    if T_total is None:
        T_total = max(max(s['frame_indices']) for s in segments) + 1

    accel_sum = np.zeros(T_total, dtype=np.float64)
    accel_cnt = np.zeros(T_total, dtype=np.int64)

    for seg in segments:
        coords = seg['coords']       # (T_seg, N, 3)
        visibs = seg['visibs']       # (T_seg, N)
        fi = seg['frame_indices']
        mov = seg['is_moving']       # (N,)
        T_seg, N_kp = visibs.shape

        # Project 3-D world coords to (u, v, z) through the first
        # frame's camera (fixed viewpoint → camera motion factored out).
        camera_view_0 = {
            'K': seg['intrs'][0],
            'c2w': np.linalg.inv(seg['extrs'][0]),
        }
        tracks_uvz = project_tracks_3d_to_3d(
            tracks3d=coords,
            camera_views=[camera_view_0] * T_seg,
        )                                                   # (T_seg, N, 3)

        # Velocity in (u, v, z) space: (T_seg-1, N, 3)
        vel = np.diff(tracks_uvz, axis=0)
        # Down-weight depth component so u/v pixel motion dominates
        vel[:, :, 2] *= depth_weight

        # Acceleration vector: difference of consecutive velocities.
        # (T_seg-2, N, 3)
        acc_vec = vel[1:] - vel[:-1]
        # Acceleration magnitude: (T_seg-2, N)
        accel = np.linalg.norm(acc_vec, axis=2)

        # --- Margin-based visibility filter ---
        # For accel[i] (centered at frame i+1), require that the
        # keypoint is continuously visible for at least `margin` frames
        # on each side.
        run_before = np.zeros_like(visibs, dtype=np.int32)
        run_before[0] = visibs[0].astype(np.int32)
        for t in range(1, T_seg):
            run_before[t] = np.where(visibs[t],
                                     run_before[t - 1] + 1, 0)

        run_after = np.zeros_like(visibs, dtype=np.int32)
        run_after[-1] = visibs[-1].astype(np.int32)
        for t in range(T_seg - 2, -1, -1):
            run_after[t] = np.where(visibs[t],
                                    run_after[t + 1] + 1, 0)

        min_run = margin + 1
        for i in range(accel.shape[0]):
            c = i + 1  # center frame
            margin_ok = ((run_before[c] >= min_run) &
                         (run_after[c] >= min_run))     # (N,)
            vis_mask = margin_ok & mov
            if not vis_mask.any():
                continue
            global_t = fi[c]
            accel_sum[global_t] += accel[i, vis_mask].sum()
            accel_cnt[global_t] += vis_mask.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        accel_per_frame = np.where(
            accel_cnt > 0,
            accel_sum / accel_cnt,
            0.0,
        ).astype(np.float32)

    return accel_per_frame


# ---------------------------------------------------------------------------
#  Reproject peak-frame 3D tracks to every frame's camera POV
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  Debug video
# ---------------------------------------------------------------------------

def _cluster_color_bgr(cluster_id):
    """Deterministic, perceptually-distinct colour for a cluster (BGR)."""
    hue = (int(cluster_id) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def save_debug_video(
    video_tensor,
    query_frame_results,
    is_moving_per_frame,
    cluster_ids_per_frame,
    output_path,
    motion_length_threshold=3.0,
    fps=10,
):
    """Render an MP4 with tracked keypoints overlaid using per-frame camera POV.

    Colour scheme (per peak-frame segment):
      * Dot colour: cluster colour if total traversal >= threshold, else grey.
      * Trail before the peak frame: green.
      * Trail after  the peak frame: red.
      * Trail colour overridden to grey if total traversal < threshold.
    """
    # video_tensor: (T, C, H, W) float [0,1]
    T = video_tensor.shape[0]
    video_np = (video_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    H, W = video_np.shape[1], video_np.shape[2]

    GREEN_BGR = (0, 200, 0)
    RED_BGR = (0, 0, 255)
    GREY_BGR = (160, 160, 160)
    TRAIL_LEN = 8

    # ------------------------------------------------------------------
    #  Build per-segment interpolated tracks in per-frame camera POV
    # ------------------------------------------------------------------
    segment_data = {}
    for peak_frame, data in query_frame_results.items():
        coords_np = data['coords'].cpu().numpy() if torch.is_tensor(data['coords']) else data['coords']
        visibs_np = data['visibs'].cpu().numpy() if torch.is_tensor(data['visibs']) else data['visibs']
        intrs = data['intrinsics_segment'].cpu().numpy() if torch.is_tensor(data['intrinsics_segment']) else data['intrinsics_segment']
        extrs = data['extrinsics_segment'].cpu().numpy() if torch.is_tensor(data['extrinsics_segment']) else data['extrinsics_segment']

        T_seg, N_pts = coords_np.shape[:2]

        frame_indices = data.get(
            'frame_indices',
            list(range(peak_frame, peak_frame + T_seg)),
        )

        # Project each timestep through its own camera (per-frame POV)
        camera_views = []
        for t_idx in range(T_seg):
            camera_views.append({
                'K': intrs[t_idx],
                'c2w': np.linalg.inv(extrs[t_idx]),
                'height': H, 'width': W,
            })
        tracks_2d = project_tracks_3d_to_2d(
            tracks3d=coords_np,
            camera_views=camera_views,
        )  # (T_seg, N_pts, 2)

        # Interpolate to fill every video frame in the segment range
        fi = np.array(frame_indices, dtype=float)
        all_t = np.arange(frame_indices[0], frame_indices[-1] + 1)

        interp_tracks = np.zeros((len(all_t), N_pts, 2))
        interp_vis = np.zeros((len(all_t), N_pts), dtype=bool)
        for p in range(N_pts):
            for d in range(2):
                interp_tracks[:, p, d] = np.interp(all_t, fi, tracks_2d[:, p, d])
            interp_vis[:, p] = np.interp(all_t, fi, visibs_np[:, p].astype(float)) > 0.5

        # Compute total 2D traversal length per point
        total_traversal = np.zeros(N_pts, dtype=np.float32)
        for p in range(N_pts):
            for tt in range(len(all_t) - 1):
                if interp_vis[tt, p] and interp_vis[tt + 1, p]:
                    total_traversal[p] += np.linalg.norm(
                        interp_tracks[tt + 1, p] - interp_tracks[tt, p])

        segment_data[peak_frame] = {
            'tracks_2d': interp_tracks,
            'visibs': interp_vis,
            'frame_range': (frame_indices[0], frame_indices[-1]),
            'cluster_ids': cluster_ids_per_frame.get(
                peak_frame, np.zeros(N_pts, dtype=int)),
            'total_traversal': total_traversal,
            'peak_frame': peak_frame,
        }

    # ------------------------------------------------------------------
    #  Render frames
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for t in range(T):
        frame_bgr = cv2.cvtColor(video_np[t].copy(), cv2.COLOR_RGB2BGR)

        for _pf, sd in segment_data.items():
            fstart, fend = sd['frame_range']
            if t < fstart or t > fend:
                continue
            lt = t - fstart
            trk = sd['tracks_2d']
            vis = sd['visibs']
            cids = sd['cluster_ids']
            traversal = sd['total_traversal']
            pf = sd['peak_frame']

            for p in range(trk.shape[1]):
                if not vis[lt, p]:
                    continue
                x, y = int(trk[lt, p, 0]), int(trk[lt, p, 1])
                if not (0 <= x < W and 0 <= y < H):
                    continue

                is_short = traversal[p] < motion_length_threshold

                # Dot colour: grey if short traversal, else cluster colour
                dot_color = GREY_BGR if is_short else _cluster_color_bgr(cids[p])
                cv2.circle(frame_bgr, (x, y), 3, dot_color, -1)

                # Trail
                for tt in range(max(0, lt - TRAIL_LEN), lt):
                    if vis[tt, p] and vis[tt + 1, p]:
                        x1, y1 = int(trk[tt, p, 0]), int(trk[tt, p, 1])
                        x2, y2 = int(trk[tt + 1, p, 0]), int(trk[tt + 1, p, 1])
                        if (0 <= x1 < W and 0 <= y1 < H
                                and 0 <= x2 < W and 0 <= y2 < H):
                            # Green before peak, red after, grey if short
                            abs_tt = fstart + tt
                            if is_short:
                                trail_color = GREY_BGR
                            elif abs_tt < pf:
                                trail_color = GREEN_BGR
                            else:
                                trail_color = RED_BGR
                            cv2.line(frame_bgr, (x1, y1), (x2, y2), trail_color, 1)

        writer.write(frame_bgr)

    writer.release()
    logger.info(f"Debug video saved: {output_path}")


def _chunk_color_bgr(chunk_idx, n_chunks):
    """Distinct colour per registration chunk (BGR).

    Uses evenly-spaced hues so neighbouring chunks are visually different.
    """
    if n_chunks <= 1:
        hue = 0.5  # cyan
    else:
        hue = (chunk_idx / n_chunks) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def _draw_triangle(img, center, size, color, thickness=-1):
    """Draw a filled (or outlined) equilateral triangle centered at (cx, cy)."""
    cx, cy = center
    h = int(size * 0.866)  # sqrt(3)/2
    pts = np.array([
        [cx, cy - size],
        [cx - h, cy + size // 2],
        [cx + h, cy + size // 2],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color) if thickness < 0 else cv2.polylines(img, [pts], True, color, thickness)


def save_debug_video_fixed_pov(
    video_tensor,
    query_frame_results,
    cluster_ids_per_frame,
    output_path,
    is_moving_per_frame=None,
    chunk_size=None,
    trail_len_before=16,
    trail_len_after=16,
    fps=10,
):
    """Render an MP4 where each frame shows past/future traces projected
    entirely through **that frame's own camera**.

    Because every 3D point is projected through a single camera, static
    objects' traces remain fixed in the image even when the camera moves.

    Visual encoding:
      * **Colour** = chunk colour for moving keypoints, gray for non-moving.
      * Past trail  (frames before current):  **green**
      * Future trail (frames after  current):  **red**
    """
    T = video_tensor.shape[0]
    video_np = (video_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    H, W = video_np.shape[1], video_np.shape[2]

    GREEN_BGR = (0, 200, 0)
    RED_BGR = (0, 0, 255)

    # Determine number of chunks for colour assignment
    if chunk_size and chunk_size > 0:
        n_chunks = math.ceil(T / chunk_size)
    else:
        n_chunks = 1

    # ------------------------------------------------------------------
    #  Pre-extract per-segment numpy arrays
    # ------------------------------------------------------------------
    segments = []
    for peak_frame, data in query_frame_results.items():
        coords_np = data['coords'].cpu().numpy() if torch.is_tensor(data['coords']) else data['coords']
        visibs_np = data['visibs'].cpu().numpy() if torch.is_tensor(data['visibs']) else data['visibs']
        intrs = data['intrinsics_segment'].cpu().numpy() if torch.is_tensor(data['intrinsics_segment']) else data['intrinsics_segment']
        extrs = data['extrinsics_segment'].cpu().numpy() if torch.is_tensor(data['extrinsics_segment']) else data['extrinsics_segment']
        frame_indices = data.get('frame_indices', list(range(peak_frame, peak_frame + coords_np.shape[0])))
        cids = cluster_ids_per_frame.get(peak_frame, np.zeros(coords_np.shape[1], dtype=int))
        is_moving = (is_moving_per_frame or {}).get(peak_frame, np.zeros(coords_np.shape[1], dtype=bool))

        # Build O(1) lookup for frame index → segment-relative index
        fi_to_rel = {f: i for i, f in enumerate(frame_indices)}

        # Registration chunk index (which chunk discovered this group)
        if chunk_size and chunk_size > 0:
            reg_chunk = frame_indices[0] // chunk_size
        else:
            reg_chunk = 0

        segments.append({
            'coords': coords_np,        # (T_seg, N, 3)
            'visibs': visibs_np,         # (T_seg, N)
            'intrs': intrs,              # (T_seg, 3, 3)
            'extrs': extrs,              # (T_seg, 4, 4)
            'frame_indices': frame_indices,
            'fi_to_rel': fi_to_rel,
            'cluster_ids': cids,
            'is_moving': np.asarray(is_moving, dtype=bool),
            'peak_frame': peak_frame,
            'reg_chunk': reg_chunk,
        })

    # ------------------------------------------------------------------
    #  Pre-compute per-frame acceleration for overlay
    # ------------------------------------------------------------------
    accel = compute_per_frame_acceleration(
        query_frame_results,
        is_moving_per_frame=is_moving_per_frame,
        T_total=T,
    )
    accel_max = accel.max() if accel.max() > 0 else 1.0

    # ------------------------------------------------------------------
    #  Render
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for t in range(T):
        frame_bgr = cv2.cvtColor(video_np[t].copy(), cv2.COLOR_RGB2BGR)

        # Draw 16x16 DINO patch grid
        GRID_COLOR = (80, 80, 80)  # dark grey
        for gi in range(1, 16):
            x = int(gi * W / 16)
            cv2.line(frame_bgr, (x, 0), (x, H - 1), GRID_COLOR, 1)
            y = int(gi * H / 16)
            cv2.line(frame_bgr, (0, y), (W - 1, y), GRID_COLOR, 1)

        # Overlay acceleration bar + text
        ac_val = float(accel[t])
        ac_norm = ac_val / accel_max  # 0-1
        bar_w = int(ac_norm * (W // 3))
        # Color: green(low) → yellow → red(high)
        bar_r = int(min(255, ac_norm * 2 * 255))
        bar_g = int(min(255, (1 - ac_norm) * 2 * 255))
        bar_color = (0, bar_g, bar_r)  # BGR
        cv2.rectangle(frame_bgr, (5, 5), (5 + bar_w, 25), bar_color, -1)
        cv2.rectangle(frame_bgr, (5, 5), (5 + W // 3, 25), (200, 200, 200), 1)
        cv2.putText(frame_bgr, f"accel: {ac_val:.4f}",
                    (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        for seg in segments:
            if t not in seg['fi_to_rel']:
                continue
            t_rel = seg['fi_to_rel'][t]
            T_seg = seg['coords'].shape[0]
            N_pts = seg['coords'].shape[1]

            # Project ALL timesteps through frame t's camera
            camera_view_t = {
                'K': seg['intrs'][t_rel],
                'c2w': np.linalg.inv(seg['extrs'][t_rel]),
                'height': H, 'width': W,
            }
            all_tracks_2d = project_tracks_3d_to_2d(
                tracks3d=seg['coords'],
                camera_views=[camera_view_t] * T_seg,
            )  # (T_seg, N, 2)

            chunk_color = _chunk_color_bgr(seg['reg_chunk'], n_chunks)

            # Determine trail window in segment-relative indices
            before_start = max(0, t_rel - trail_len_before)
            after_end = min(T_seg, t_rel + trail_len_after + 1)

            GRAY_BGR = (140, 140, 140)

            for p in range(N_pts):
                if not seg['visibs'][t_rel, p]:
                    continue

                is_mov = seg['is_moving'][p] if p < len(seg['is_moving']) else False

                x, y = int(all_tracks_2d[t_rel, p, 0]), int(all_tracks_2d[t_rel, p, 1])
                if not (0 <= x < W and 0 <= y < H):
                    continue

                if not is_mov:
                    # Non-moving / background keypoints: gray dot, no trails
                    cv2.circle(frame_bgr, (x, y), 2, GRAY_BGR, -1)
                    continue

                cv2.circle(frame_bgr, (x, y), 3, chunk_color, -1)

                # Past trail (before current frame) — green
                for tt in range(before_start, t_rel):
                    if seg['visibs'][tt, p] and seg['visibs'][tt + 1, p]:
                        x1, y1 = int(all_tracks_2d[tt, p, 0]), int(all_tracks_2d[tt, p, 1])
                        x2, y2 = int(all_tracks_2d[tt + 1, p, 0]), int(all_tracks_2d[tt + 1, p, 1])
                        if (0 <= x1 < W and 0 <= y1 < H
                                and 0 <= x2 < W and 0 <= y2 < H):
                            cv2.line(frame_bgr, (x1, y1), (x2, y2), GREEN_BGR, 1)

                # Future trail (after current frame) — red
                for tt in range(t_rel, after_end - 1):
                    if seg['visibs'][tt, p] and seg['visibs'][tt + 1, p]:
                        x1, y1 = int(all_tracks_2d[tt, p, 0]), int(all_tracks_2d[tt, p, 1])
                        x2, y2 = int(all_tracks_2d[tt + 1, p, 0]), int(all_tracks_2d[tt + 1, p, 1])
                        if (0 <= x1 < W and 0 <= y1 < H
                                and 0 <= x2 < W and 0 <= y2 < H):
                            cv2.line(frame_bgr, (x1, y1), (x2, y2), RED_BGR, 1)

        writer.write(frame_bgr)

    writer.release()
    logger.info(f"Fixed-POV debug video saved: {output_path}")


# ---------------------------------------------------------------------------
#  Structured data saving
# ---------------------------------------------------------------------------

def save_structured_data(
    video_name,
    output_dir,
    video_tensor,
    depths,
    coords,
    visibs,
    intrinsics,
    extrinsics,
    query_points_per_frame,
    original_filenames,
    query_frame_results=None,
    future_len: int = 128,
    history_len: int = 32,
    cluster_ids_per_frame=None,
    is_moving_per_frame=None,
    max_disp_per_frame=None,
    video_tensor_full=None,
    video_tensor_orig=None,
):
    """Save data in the structured format.

    If ``video_tensor_orig`` is supplied, it is assumed to be the original
    (pre-VGGT-resize) video tensor. In that case the saved images, depth,
    intrinsics and per-sample track coordinates are all converted back to
    the original resolution so downstream tools never see the bicubic
    upsampling artifacts introduced by VGGT's 518-px preprocessing.
    """

    # Create output directories
    video_output_dir = os.path.join(output_dir, video_name)
    samples_dir = os.path.join(video_output_dir, "samples")

    # Save structured data in the new format
    for dir_path in [video_output_dir, samples_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # ------------------------------------------------------------------
    #  Resolution bookkeeping: the pipeline runs in the 518-wide
    #  "processed" space that VGGT expects, but we want disk artifacts
    #  at the original capture resolution. Scale factors go from
    #  processed-space pixel coords to original-space pixel coords.
    # ------------------------------------------------------------------
    proc_H = proc_W = 0
    if video_tensor_full is not None:
        proc_H = int(video_tensor_full.shape[-2])
        proc_W = int(video_tensor_full.shape[-1])
    if video_tensor_orig is not None:
        orig_H = int(video_tensor_orig.shape[-2])
        orig_W = int(video_tensor_orig.shape[-1])
    else:
        orig_H, orig_W = proc_H, proc_W
    sx = (orig_W / proc_W) if proc_W else 1.0
    sy = (orig_H / proc_H) if proc_H else 1.0

    # Save ALL frames' images and depth as single .npy files
    if video_tensor_full is not None:
        # Image source: prefer the original-resolution tensor.
        img_tensor_for_save = video_tensor_orig if video_tensor_orig is not None else video_tensor_full
        full_video_np = (img_tensor_for_save.permute(0, 2, 3, 1).cpu().numpy()
                         * 255).astype(np.uint8)
        T_full = full_video_np.shape[0]

        # Save images as a single .npy: (T, H, W, 3) uint8
        images_npy_path = os.path.join(video_output_dir, "images.npy")
        np.save(images_npy_path, full_video_np)

        # Use the full-video depth tensor passed in directly (do not duplicate
        # it inside per-group entries — that caused host-RAM OOM on long videos).
        full_depths_np = None
        if depths is not None:
            if torch.is_tensor(depths):
                if video_tensor_orig is not None and (depths.shape[-2:] != (orig_H, orig_W)):
                    # Bilinear resize depth from processed space back to the
                    # original capture resolution. Depth values are metric,
                    # so no rescaling of the z-axis is needed.
                    full_depths_np = torch.nn.functional.interpolate(
                        depths.unsqueeze(1).float(),
                        size=(orig_H, orig_W),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1).cpu().numpy()
                else:
                    full_depths_np = depths.cpu().numpy()
            else:
                full_depths_np = depths

        # Save depth as a single .npy: (T, H, W) float16 meters
        if full_depths_np is not None:
            depth_npy_path = os.path.join(video_output_dir, "depth.npy")
            np.save(depth_npy_path, full_depths_np.astype(np.float16))

        logger.info(f"Saved {T_full} frames to images.npy and depth.npy")

    # ------------------------------------------------------------------
    #  Save per-video camera parameters (intrinsics + extrinsics) to a
    #  single cameras.npz, so downstream tools can unproject/reproject
    #  between any two frames of the same video without re-running infer.
    # ------------------------------------------------------------------
    if intrinsics is not None and extrinsics is not None:
        intr_np = intrinsics.cpu().numpy() if torch.is_tensor(intrinsics) else intrinsics
        extr_np = extrinsics.cpu().numpy() if torch.is_tensor(extrinsics) else extrinsics
        # Scale intrinsics from processed (518-wide) space back to the
        # original capture resolution so they match the saved PNGs.
        intr_save = intr_np.astype(np.float32).copy()
        intr_save[..., 0, 0] *= sx
        intr_save[..., 0, 2] *= sx
        intr_save[..., 1, 1] *= sy
        intr_save[..., 1, 2] *= sy
        cameras_path = os.path.join(video_output_dir, "cameras.npz")
        np.savez(
            cameras_path,
            intrinsics=intr_save,                    # (T, 3, 3) in orig-res space
            extrinsics=extr_np.astype(np.float32),   # (T, 4, 4) world-to-camera
            height=np.int32(orig_H),
            width=np.int32(orig_W),
        )
        logger.info(
            f"Saved camera parameters for {intr_save.shape[0]} frames to {cameras_path}"
        )

    # ------------------------------------------------------------------
    #  Save per-frame acceleration (from 3D world-frame trajectories)
    # ------------------------------------------------------------------
    if query_frame_results is not None and video_tensor_full is not None:
        T_full = video_tensor_full.shape[0]
        accel = compute_per_frame_acceleration(
            query_frame_results,
            is_moving_per_frame=is_moving_per_frame,
            T_total=T_full,
        )
        accel_path = os.path.join(video_output_dir, "acceleration.npz")
        np.savez(accel_path, acceleration=accel)  # (T,) float32
        logger.info(
            f"Saved per-frame acceleration ({T_full} frames) to {accel_path}  "
            f"mean={accel.mean():.4f}  max={accel.max():.4f}"
        )

    # ------------------------------------------------------------------
    #  Save movement statistics (unique keypoints above displacement thresholds,
    #  plus per-step camera displacement / rotation between saved frames).
    # ------------------------------------------------------------------
    stats_to_save = {}
    log_parts = []

    if max_disp_per_frame:
        norm_scale = max(orig_W, orig_H, 1)
        total_unique_kps = 0
        n_above_005 = 0
        n_above_010 = 0
        for _pf, md in max_disp_per_frame.items():
            disp_norm = md / norm_scale
            total_unique_kps += len(md)
            n_above_005 += int((disp_norm > 0.05).sum())
            n_above_010 += int((disp_norm > 0.10).sum())

        stats_to_save.update({
            "total_unique_kps": np.array(total_unique_kps),
            "n_above_005": np.array(n_above_005),
            "n_above_010": np.array(n_above_010),
        })
        log_parts.append(
            f"total={total_unique_kps}, >0.05={n_above_005}, >0.10={n_above_010}"
        )

    # Per-step camera motion between consecutive saved frames.
    # extr_np is world-to-camera (T, 4, 4); camera center C = -R^T @ t.
    if intrinsics is not None and extrinsics is not None and extr_np.shape[0] >= 2:
        R = extr_np[:, :3, :3].astype(np.float64)          # (T, 3, 3)
        t = extr_np[:, :3, 3].astype(np.float64)           # (T, 3)
        C = -np.einsum('tji,tj->ti', R, t)                 # (T, 3) camera centers in world
        trans_step = np.linalg.norm(np.diff(C, axis=0), axis=1)  # (T-1,)

        # Relative rotation between consecutive frames: R_rel = R_{t+1} @ R_t^T
        R_rel = np.einsum('tij,tkj->tik', R[1:], R[:-1])
        cos_ang = (np.trace(R_rel, axis1=1, axis2=2) - 1.0) * 0.5
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        rot_step = np.arccos(cos_ang)                      # radians, (T-1,)

        stats_to_save.update({
            "cam_trans_step": trans_step.astype(np.float32),       # world-unit per step
            "cam_rot_step_rad": rot_step.astype(np.float32),       # radians per step
            "cam_trans_total": np.float32(trans_step.sum()),
            "cam_rot_total_rad": np.float32(rot_step.sum()),
            "cam_trans_mean": np.float32(trans_step.mean()),
            "cam_rot_mean_rad": np.float32(rot_step.mean()),
            "cam_trans_max": np.float32(trans_step.max()),
            "cam_rot_max_rad": np.float32(rot_step.max()),
        })
        log_parts.append(
            f"cam_trans total={trans_step.sum():.3f} mean={trans_step.mean():.4f} max={trans_step.max():.4f}; "
            f"cam_rot total={np.degrees(rot_step.sum()):.1f}deg "
            f"mean={np.degrees(rot_step.mean()):.3f}deg max={np.degrees(rot_step.max()):.3f}deg"
        )

    if stats_to_save:
        stats_path = os.path.join(video_output_dir, "movement_statistics.npz")
        np.savez(stats_path, **stats_to_save)
        logger.info(
            f"Saved movement statistics to {stats_path}  " + " | ".join(log_parts)
        )

    # ------------------------------------------------------------------
    #  Save per-frame samples: for every frame, project full trajectory
    #  through that frame's camera (matches debug_fixed_pov.mp4).
    #  traj = future trace, traj_history = past trace.
    # ------------------------------------------------------------------
    if query_frame_results is not None and video_tensor_full is not None:
        T_full = video_tensor_full.shape[0]
        # Per-sample tracks are saved in the original capture resolution, so
        # we use orig_H/orig_W as the reference dimensions for retargeting
        # and normalization below. The per-frame segment intrinsics still
        # live in processed space — we scale the projected pixel coords to
        # orig-space right after projection.
        H_full, W_full = orig_H, orig_W

        # Pre-extract numpy arrays from all peak-frame segments
        segments = []
        for peak_frame, fdata in query_frame_results.items():
            coords_np = fdata['coords'].cpu().numpy() if torch.is_tensor(fdata['coords']) else fdata['coords']
            visibs_np = fdata['visibs'].cpu().numpy() if torch.is_tensor(fdata['visibs']) else fdata['visibs']
            intrs = fdata['intrinsics_segment'].cpu().numpy() if torch.is_tensor(fdata['intrinsics_segment']) else fdata['intrinsics_segment']
            extrs = fdata['extrinsics_segment'].cpu().numpy() if torch.is_tensor(fdata['extrinsics_segment']) else fdata['extrinsics_segment']
            fi = fdata.get('frame_indices', list(range(coords_np.shape[0])))
            cids = (cluster_ids_per_frame or {}).get(
                peak_frame, np.zeros(coords_np.shape[1], dtype=np.int32))
            is_mov = (is_moving_per_frame or {}).get(
                peak_frame, np.ones(coords_np.shape[1], dtype=bool))
            # O(1) lookup for frame index → segment-relative index
            fi_to_rel = {f: i for i, f in enumerate(fi)}

            segments.append({
                'coords': coords_np,       # (T_seg, N, 3)
                'visibs': visibs_np,        # (T_seg, N)
                'intrs': intrs,
                'extrs': extrs,
                'frame_indices': fi,
                'fi_to_rel': fi_to_rel,
                'cluster_ids': np.asarray(cids, dtype=np.int32),
                'is_moving': np.asarray(is_mov, dtype=bool),
            })

        # Accumulators for unified .npy saving
        acc_frame_indices = []      # which frame t each block belongs to
        acc_keypoints = []          # (N_t, 2) per frame
        acc_cluster_ids = []
        acc_is_moving = []
        acc_visibs = []
        acc_traj = []               # (N_t, future_len, 3) per frame
        acc_traj_history = []       # (N_t, history_len, 3)
        acc_valid_steps = []        # (N_t, future_len)
        acc_valid_steps_history = []
        acc_raw_traj = []           # (N_t, future_len, 3) padded
        acc_raw_traj_history = []   # (N_t, history_len, 3) padded
        acc_raw_valid_steps = []
        acc_raw_valid_steps_history = []
        offsets = [0]               # offsets[i] = start index of frame i in concatenated arrays

        saved_count = 0
        for t in range(T_full):
            per_seg_traj_future = []      # list of (N_seg, future_len, 3)
            per_seg_traj_history = []
            per_seg_valid_future = []     # list of (N_seg, future_len)
            per_seg_valid_history = []
            per_seg_raw_future = []       # list of (N_seg, T_future_seg, 3) — pre-retarget
            per_seg_raw_history = []      # list of (N_seg, T_history_seg, 3) — pre-retarget
            all_keypoints = []
            all_cluster_ids = []
            all_is_moving = []
            all_visibs = []

            for seg in segments:
                if t not in seg['fi_to_rel']:
                    continue
                t_rel = seg['fi_to_rel'][t]
                T_seg = seg['coords'].shape[0]

                # Project ALL timesteps through frame t's camera
                camera_view_t = {
                    'K': seg['intrs'][t_rel],
                    'c2w': np.linalg.inv(seg['extrs'][t_rel]),
                    'height': H_full, 'width': W_full,
                }
                projected = project_tracks_3d_to_3d(
                    tracks3d=seg['coords'],
                    camera_views=[camera_view_t] * T_seg,
                )  # (T_seg, N, 3) = (x_pixel, y_pixel, z_depth) in proc space
                # Scale pixel coords back to the original capture resolution.
                if sx != 1.0 or sy != 1.0:
                    projected[..., 0] *= sx
                    projected[..., 1] *= sy

                # Keypoints = position at current frame t
                all_keypoints.append(projected[t_rel, :, :2])  # (N, 2)

                # Future: from current frame onward → (N_seg, T_future, 3)
                future = projected[t_rel:].transpose(1, 0, 2)
                # History: from current frame backward (reversed) → (N_seg, T_past, 3)
                history = projected[:t_rel + 1][::-1].copy().transpose(1, 0, 2)

                N_seg = future.shape[0]

                # Retarget each segment INDEPENDENTLY so that this segment's
                # tracks share a true lifespan (no constant-hold padding from
                # other segments). retarget_trajectories already initializes
                # output buffers with -inf in the trailing slots.
                if future.shape[1] >= 2:
                    future_ret, future_mask_1d = retarget_trajectories(
                        future, max_length=future_len,
                        image_width=W_full, image_height=H_full)
                else:
                    future_ret = np.full((N_seg, future_len, 3), -np.inf,
                                         dtype=future.dtype)
                    future_mask_1d = np.zeros(future_len, dtype=bool)
                    if future.shape[1] == 1:
                        future_ret[:, 0, :] = future[:, 0, :]
                        future_mask_1d[0] = True

                if history.shape[1] >= 2:
                    history_ret, history_mask_1d = retarget_trajectories(
                        history, max_length=history_len,
                        image_width=W_full, image_height=H_full)
                else:
                    history_ret = np.full((N_seg, history_len, 3), -np.inf,
                                          dtype=history.dtype)
                    history_mask_1d = np.zeros(history_len, dtype=bool)
                    if history.shape[1] == 1:
                        history_ret[:, 0, :] = history[:, 0, :]
                        history_mask_1d[0] = True

                # Broadcast each segment's mask along N
                future_mask_2d = np.broadcast_to(
                    future_mask_1d, (N_seg, future_len)).copy()
                history_mask_2d = np.broadcast_to(
                    history_mask_1d, (N_seg, history_len)).copy()

                per_seg_traj_future.append(future_ret.astype(np.float32))
                per_seg_traj_history.append(history_ret.astype(np.float32))
                per_seg_valid_future.append(future_mask_2d)
                per_seg_valid_history.append(history_mask_2d)

                # Stash the pre-retarget per-frame trajectories. Each slot
                # along the time axis corresponds 1-1 to a real frame index
                # (frame t + k for future, frame t - k for history).
                per_seg_raw_future.append(future.astype(np.float32))
                per_seg_raw_history.append(history.astype(np.float32))

                all_cluster_ids.append(seg['cluster_ids'])
                all_is_moving.append(seg['is_moving'])
                all_visibs.append(seg['visibs'][t_rel])

            if not per_seg_traj_future:
                continue

            # Concatenate across peak-frame segments along the N axis
            keypoints = np.concatenate(all_keypoints, axis=0).astype(np.float32)
            cluster_ids = np.concatenate(all_cluster_ids, axis=0)
            is_moving = np.concatenate(all_is_moving, axis=0)
            visibs = np.concatenate(all_visibs, axis=0)
            traj_ret = np.concatenate(per_seg_traj_future, axis=0)
            traj_hist_ret = np.concatenate(per_seg_traj_history, axis=0)
            valid_future = np.concatenate(per_seg_valid_future, axis=0)
            valid_history = np.concatenate(per_seg_valid_history, axis=0)

            # Filter out non-moving keypoints to save memory
            mov_mask = is_moving.astype(bool)
            if not mov_mask.all():
                keypoints = keypoints[mov_mask]
                cluster_ids = cluster_ids[mov_mask]
                visibs = visibs[mov_mask]
                traj_ret = traj_ret[mov_mask]
                traj_hist_ret = traj_hist_ret[mov_mask]
                valid_future = valid_future[mov_mask]
                valid_history = valid_history[mov_mask]
                # Filter raw segments: apply mask per-segment
                filtered_raw_future, filtered_raw_history = [], []
                offset = 0
                for fut_seg, hist_seg in zip(per_seg_raw_future, per_seg_raw_history):
                    n = fut_seg.shape[0]
                    seg_mask = mov_mask[offset:offset + n]
                    filtered_raw_future.append(fut_seg[seg_mask])
                    filtered_raw_history.append(hist_seg[seg_mask])
                    offset += n
                per_seg_raw_future = filtered_raw_future
                per_seg_raw_history = filtered_raw_history
                is_moving = is_moving[mov_mask]

            if len(keypoints) == 0:
                continue

            # Filter out off-screen keypoints.  Even though trajectories
            # are no longer clipped (off-screen motion is preserved), a
            # keypoint that is outside the image at the current frame would
            # produce a trace with no visible marker — appearing as a
            # disconnected line in visualizations.
            on_screen = (
                (keypoints[:, 0] >= 0) & (keypoints[:, 0] <= W_full) &
                (keypoints[:, 1] >= 0) & (keypoints[:, 1] <= H_full)
            )
            if not on_screen.all():
                keypoints = keypoints[on_screen]
                cluster_ids = cluster_ids[on_screen]
                is_moving = is_moving[on_screen]
                visibs = visibs[on_screen]
                traj_ret = traj_ret[on_screen]
                traj_hist_ret = traj_hist_ret[on_screen]
                valid_future = valid_future[on_screen]
                valid_history = valid_history[on_screen]
                # Filter raw segments
                filtered_raw_future, filtered_raw_history = [], []
                offset = 0
                for fut_seg, hist_seg in zip(per_seg_raw_future, per_seg_raw_history):
                    n = fut_seg.shape[0]
                    seg_mask = on_screen[offset:offset + n]
                    filtered_raw_future.append(fut_seg[seg_mask])
                    filtered_raw_history.append(hist_seg[seg_mask])
                    offset += n
                per_seg_raw_future = filtered_raw_future
                per_seg_raw_history = filtered_raw_history

            if len(keypoints) == 0:
                continue

            # Skip frames with no meaningful future trajectory (< 2 valid
            # steps across all tracks).  These occur near the end of the
            # video where there is no future movement to retarget.
            max_future_steps = int(valid_future.sum(axis=1).max())
            if max_future_steps < 2:
                continue

            # ----- Assemble raw (pre-retarget) trajectories -----------------
            # Different segments have different remaining lifespans, so we
            # right-pad each one with -inf to a common time length and then
            # concatenate along the N axis. For unified saving, we pad all
            # raw arrays to future_len / history_len so they share a common
            # time dimension across all frames.
            raw_fut_chunks, raw_hist_chunks = [], []
            raw_fut_mask_chunks, raw_hist_mask_chunks = [], []
            for fut_seg, hist_seg in zip(per_seg_raw_future, per_seg_raw_history):
                N_seg = fut_seg.shape[0]
                T_fut_seg = min(fut_seg.shape[1], future_len)
                T_hist_seg = min(hist_seg.shape[1], history_len)

                padded_fut = np.full((N_seg, future_len, 3), -np.inf, dtype=np.float32)
                padded_fut[:, :T_fut_seg, :] = fut_seg[:, :T_fut_seg, :]
                raw_fut_chunks.append(padded_fut)

                padded_hist = np.full((N_seg, history_len, 3), -np.inf, dtype=np.float32)
                padded_hist[:, :T_hist_seg, :] = hist_seg[:, :T_hist_seg, :]
                raw_hist_chunks.append(padded_hist)

                m_fut = np.zeros((N_seg, future_len), dtype=bool)
                m_fut[:, :T_fut_seg] = True
                raw_fut_mask_chunks.append(m_fut)

                m_hist = np.zeros((N_seg, history_len), dtype=bool)
                m_hist[:, :T_hist_seg] = True
                raw_hist_mask_chunks.append(m_hist)

            raw_traj = np.concatenate(raw_fut_chunks, axis=0)
            raw_traj_history = np.concatenate(raw_hist_chunks, axis=0)
            raw_valid_steps = np.concatenate(raw_fut_mask_chunks, axis=0)
            raw_valid_steps_history = np.concatenate(raw_hist_mask_chunks, axis=0)

            # Accumulate for unified .npy saving
            N_t = len(keypoints)
            acc_frame_indices.append(t)
            acc_keypoints.append(keypoints.astype(np.float16))
            acc_cluster_ids.append(cluster_ids)
            acc_is_moving.append(is_moving)
            acc_visibs.append(visibs.astype(np.float16))
            acc_traj.append(traj_ret.astype(np.float16))
            acc_traj_history.append(traj_hist_ret.astype(np.float16))
            acc_valid_steps.append(valid_future)
            acc_valid_steps_history.append(valid_history)
            acc_raw_traj.append(raw_traj.astype(np.float16))
            acc_raw_traj_history.append(raw_traj_history.astype(np.float16))
            acc_raw_valid_steps.append(raw_valid_steps)
            acc_raw_valid_steps_history.append(raw_valid_steps_history)
            offsets.append(offsets[-1] + N_t)
            saved_count += 1

        # ----- Write unified .npy files to samples/ -----------------------
        if saved_count > 0:
            np.save(os.path.join(samples_dir, "frame_indices.npy"),
                    np.array(acc_frame_indices, dtype=np.int32))
            np.save(os.path.join(samples_dir, "offsets.npy"),
                    np.array(offsets, dtype=np.int64))
            np.save(os.path.join(samples_dir, "keypoints.npy"),
                    np.concatenate(acc_keypoints, axis=0))
            np.save(os.path.join(samples_dir, "cluster_ids.npy"),
                    np.concatenate(acc_cluster_ids, axis=0))
            np.save(os.path.join(samples_dir, "is_moving.npy"),
                    np.concatenate(acc_is_moving, axis=0))
            np.save(os.path.join(samples_dir, "visibs.npy"),
                    np.concatenate(acc_visibs, axis=0))
            np.save(os.path.join(samples_dir, "traj.npy"),
                    np.concatenate(acc_traj, axis=0))
            np.save(os.path.join(samples_dir, "traj_history.npy"),
                    np.concatenate(acc_traj_history, axis=0))
            np.save(os.path.join(samples_dir, "valid_steps.npy"),
                    np.concatenate(acc_valid_steps, axis=0))
            np.save(os.path.join(samples_dir, "valid_steps_history.npy"),
                    np.concatenate(acc_valid_steps_history, axis=0))
            np.save(os.path.join(samples_dir, "raw_traj.npy"),
                    np.concatenate(acc_raw_traj, axis=0))
            np.save(os.path.join(samples_dir, "raw_traj_history.npy"),
                    np.concatenate(acc_raw_traj_history, axis=0))
            np.save(os.path.join(samples_dir, "raw_valid_steps.npy"),
                    np.concatenate(acc_raw_valid_steps, axis=0))
            np.save(os.path.join(samples_dir, "raw_valid_steps_history.npy"),
                    np.concatenate(acc_raw_valid_steps_history, axis=0))

        logger.info(
            f"Saved {saved_count} frames ({offsets[-1]} total keypoints) "
            f"as unified .npy to {samples_dir}"
        )


# ---------------------------------------------------------------------------
#  Long-video progressive tracking pipeline
# ---------------------------------------------------------------------------

def _run_tracking_on_chunk(
    model_3dtracker,
    chunk_video,
    chunk_depth,
    chunk_intrs,
    chunk_extrs,
    query_points,
    frame_H,
    frame_W,
    args,
    queries_are_3d=False,
):
    """Run TAPIP3D tracking on a single chunk.

    Args:
        chunk_video: (T, 3, H, W) preprocessed video tensor.
        chunk_depth: (T, H, W) numpy depth (will be copied internally).
        chunk_intrs: (T, 3, 3) numpy intrinsics (will be copied internally).
        chunk_extrs: (T, 4, 4) numpy world-to-camera extrinsics (will be copied).
        query_points: list of (N_i, 3-or-4) numpy arrays.
            If queries_are_3d=False: each is [t, x, y] (2D).
            If queries_are_3d=True: each is [t, wx, wy, wz] (3D world).
        frame_H, frame_W: image dimensions.
        args: parsed arguments.
        queries_are_3d: whether queries are pre-computed 3D world coords.

    Returns:
        coords: (T, N, 3) 3D world coordinates.
        visibs: (T, N) visibility.
    """
    video, depths, intrinsics, extrinsics, query_point_tensor, support_grid_size = (
        prepare_inputs(
            chunk_video,
            chunk_depth.copy(),
            chunk_intrs.copy(),
            chunk_extrs.copy(),
            query_points,
            inference_res=(frame_H, frame_W),
            support_grid_size=16,
            device=args.device,
            queries_are_3d=queries_are_3d,
        )
    )

    model_3dtracker.set_image_size((frame_H, frame_W))

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            coords, visibs = inference(
                model=model_3dtracker,
                video=video,
                depths=depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                query_point=query_point_tensor,
                num_iters=args.num_iters,
                grid_size=support_grid_size,
                bidrectional=True,
            )

    return coords, visibs


def process_long_video(
    video_path,
    depth_path,
    args,
    model_3dtracker,
    model_depth_pose,
    dino_extractor,
):
    """Process a video using the long-video progressive tracking pipeline.

    Pipeline:
        1. Load full video (with optional frame_step subsampling).
        2. Run hybrid chunked VGGT (global sparse + local dense).
        3. For each tracking chunk:
           - Chunk 0: DINO → peak frames → keypoints → TAPIP3D
           - Chunk N≥1: propagate active keypoints → DINO → proximity filter
             → track new → merge
        4. Assemble global trajectories.
        5. Motion detection.

    Returns:
        dict consumed by save_structured_data() and the debug-video writers.
    """
    logger.info(f"Processing video (long-video pipeline): {video_path}")
    _pipeline_t0 = time.time()
    _log_vram("Pipeline start")

    # ==================================================================
    #  1. Load full video
    # ==================================================================
    dataset = get_dataset(args.dataset_name) if args.dataset_name else None
    if dataset is not None and dataset.load_frames is not None:
        video_tensor, video_mask, original_filenames = dataset.load_frames(video_path, args)
        video_tensor = resize_frames(video_tensor, args.target_hw)
    else:
        video_tensor, video_mask, original_filenames = load_video_and_mask(
            video_path,
            frame_step=args.frame_step,
            target_fps=args.target_fps,
            target_hw=args.target_hw,
        )

    T_total_raw = len(video_tensor)
    logger.info(f"Loaded {T_total_raw} frames from {video_path}")
    if T_total_raw < 2:
        raise ValueError(
            f"Only {T_total_raw} frame(s) after subsampling; at least 2 are needed "
            f"(check --frame_step / --target_fps)"
        )

    # Load known depth if provided
    depth_tensor = None
    if depth_path is not None:
        depth_tensor, _, _ = load_video_and_mask(
            depth_path, None, frame_step=args.frame_step, is_depth=True,
            target_fps=args.target_fps,
            target_hw=args.target_hw,
        )
        depth_tensor[depth_tensor <= 0] = 0

    # ==================================================================
    #  2. Chunked VGGT: global sparse + local dense
    # ==================================================================
    _stage2_t0 = time.time()
    _log_vram("Before VGGT")
    vggt_chunk_size = args.chunk_size
    tracking_chunk_size = getattr(args, 'tracking_chunk_size', None) or vggt_chunk_size
    sparse_max = getattr(args, 'sparse_max', 150)
    # Clamp sparse_max to vggt_chunk_size so the sparse pass fits in the same VRAM budget
    if sparse_max > vggt_chunk_size:
        logger.warning(
            f"sparse_max ({sparse_max}) > chunk_size ({vggt_chunk_size}), "
            f"clamping to {vggt_chunk_size}"
        )
        sparse_max = vggt_chunk_size
    logger.info(f"Chunk sizes: VGGT={vggt_chunk_size}, tracking={tracking_chunk_size}, sparse_max={sparse_max}")

    # Offload tracker and DINO to CPU before VGGT runs
    _move_model_to(model_3dtracker, "cpu")
    _move_model_to(dino_extractor, "cpu")
    _move_model_to(model_depth_pose, args.device)

    (
        video_ten, depth_npy, depth_conf, extrs_c2w, intrs_npy
    ) = model_depth_pose.chunked_inference(
        video_tensor,
        chunk_size=vggt_chunk_size,
        sparse_max=sparse_max,
        known_depth=depth_tensor,
        stationary_camera=False,
        replace_with_known_depth=False,
    )

    depth_conf_npy = depth_conf.squeeze() if hasattr(depth_conf, 'squeeze') else depth_conf
    if isinstance(depth_conf_npy, torch.Tensor):
        depth_conf_npy = depth_conf_npy.cpu().numpy()

    T_total = len(video_ten)
    frame_H, frame_W = video_ten.shape[-2:]

    # Convert c2w → w2c for tracking
    extrs_w2c = np.linalg.inv(extrs_c2w)

    # Offload VGGT — no longer needed
    _move_model_to(model_depth_pose, "cpu")
    logger.info(f"[TIMER] VGGT inference took {time.time() - _stage2_t0:.1f}s")
    _log_vram("After VGGT (offloaded)")

    # ==================================================================
    #  3. Compute tracking chunk boundaries (non-overlapping)
    # ==================================================================
    n_chunks = math.ceil(T_total / tracking_chunk_size)
    chunk_ranges = []
    for i in range(n_chunks):
        start = i * tracking_chunk_size
        end = min(start + tracking_chunk_size, T_total)
        if start < end:
            chunk_ranges.append((start, end))
    # A tail chunk shorter than two DINO strides yields at most one DINO frame,
    # which breaks keypoint extraction / trajectory assembly (e.g. 361 frames
    # with tracking_chunk_size=360). Merge it into the previous chunk.
    min_tail = 2 * args.dino_stride
    if len(chunk_ranges) >= 2 and chunk_ranges[-1][1] - chunk_ranges[-1][0] < min_tail:
        chunk_ranges[-2:] = [(chunk_ranges[-2][0], chunk_ranges[-1][1])]

    logger.info(
        f"Long-video pipeline: {T_total} frames → "
        f"{len(chunk_ranges)} tracking chunk(s) of ≤{tracking_chunk_size} frames"
    )

    # ==================================================================
    #  4. Progressive tracking
    # ==================================================================
    # trajectory_groups[peak_frame_global] = {
    #     'coords_chunks': [(T_chunk, N, 3), ...],
    #     'visibs_chunks': [(T_chunk, N), ...],
    #     'cluster_ids': (N,) int array,
    #     'peak_frame_global': int,
    #     'first_global_frame': int,  # first frame of registration chunk
    # }
    trajectory_groups = {}
    # group_order: list of peak_frame_global in registration order
    group_order = []
    # active_groups: peak_frame_globals of groups to propagate (all kps, regardless of vis)
    active_groups = []

    query_points_per_frame = {}

    _stage4_t0 = time.time()
    for chunk_idx, (cs, ce) in enumerate(chunk_ranges):
        _chunk_t0 = time.time()
        T_chunk = ce - cs
        chunk_video = video_ten[cs:ce]
        chunk_depth = depth_npy[cs:ce]
        chunk_intrs = intrs_npy[cs:ce]
        chunk_extrs = extrs_w2c[cs:ce]
        chunk_frame_indices = list(range(cs, ce))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _log_vram(f"Chunk {chunk_idx} start")
        logger.info(
            f"── Tracking chunk {chunk_idx}: frames [{cs}, {ce}) "
            f"({T_chunk} frames) ──"
        )

        # ==============================================================
        #  Step A: Propagate active keypoints from previous chunk
        # ==============================================================
        if chunk_idx > 0 and len(active_groups) > 0:
            # Move tracker to GPU for propagation
            _move_model_to(dino_extractor, "cpu")
            _move_model_to(model_3dtracker, args.device)
            # Build one big batch of 3D queries for all propagated groups
            prop_queries_list = []
            prop_group_slices = []  # (peak_frame, start_idx, end_idx)
            offset = 0

            for gpf in active_groups:
                tg = trajectory_groups[gpf]
                # Last frame's 3D world coords from previous chunk
                last_coords = tg['coords_chunks'][-1][-1]  # (N, 3)
                if isinstance(last_coords, torch.Tensor):
                    last_coords = last_coords.cpu().numpy()
                last_coords = np.atleast_2d(last_coords)  # ensure (N, 3) even if N=1
                N = last_coords.shape[0]
                # Query: [t=0, wx, wy, wz]
                q = np.concatenate([
                    np.zeros((N, 1), dtype=np.float32),
                    last_coords.astype(np.float32),
                ], axis=1)
                prop_queries_list.append(q)
                prop_group_slices.append((gpf, offset, offset + N))
                offset += N

            all_prop_queries = np.concatenate(prop_queries_list, axis=0)
            logger.info(
                f"  Propagating {all_prop_queries.shape[0]} keypoints "
                f"from {len(active_groups)} group(s)"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            coords_prop, visibs_prop = _run_tracking_on_chunk(
                model_3dtracker,
                chunk_video, chunk_depth, chunk_intrs, chunk_extrs,
                [all_prop_queries],
                frame_H, frame_W, args,
                queries_are_3d=True,
            )

            # Distribute results back to groups
            coords_prop_np = coords_prop.cpu().numpy() if torch.is_tensor(coords_prop) else coords_prop
            visibs_prop_np = visibs_prop.cpu().numpy() if torch.is_tensor(visibs_prop) else visibs_prop
            # Ensure (T, N, 3) and (T, N) even when N=1
            if coords_prop_np.ndim == 2:
                coords_prop_np = coords_prop_np[:, np.newaxis, :]
            if visibs_prop_np.ndim == 1:
                visibs_prop_np = visibs_prop_np[:, np.newaxis]

            for gpf, si, ei in prop_group_slices:
                trajectory_groups[gpf]['coords_chunks'].append(coords_prop_np[:, si:ei, :])
                trajectory_groups[gpf]['visibs_chunks'].append(visibs_prop_np[:, si:ei])
                trajectory_groups[gpf]['frame_ranges'].append((cs, ce))

        _log_vram(f"Chunk {chunk_idx} after propagation")
        # ==============================================================
        #  Step B: Compute covered patch mask + DINO clustering
        # ==============================================================
        # Move DINO to GPU, tracker to CPU
        _move_model_to(model_3dtracker, "cpu")
        _move_model_to(dino_extractor, args.device)

        # Build (T_chunk, 16, 16) mask of patches already covered by tracks
        covered_patches = None
        if chunk_idx > 0 and len(active_groups) > 0:
            covered_patches = np.zeros((T_chunk, 16, 16), dtype=bool)
            for gpf in active_groups:
                tg = trajectory_groups[gpf]
                latest_coords = tg['coords_chunks'][-1]  # (T_chunk, N, 3)
                latest_visibs = tg['visibs_chunks'][-1]   # (T_chunk, N)
                for t_local in range(T_chunk):
                    vis = latest_visibs[t_local]
                    visible = vis > 0.5
                    if not np.any(visible):
                        continue
                    coords_3d = latest_coords[t_local]  # (N, 3)
                    camera_view = {
                        'K': chunk_intrs[t_local],
                        'c2w': np.linalg.inv(chunk_extrs[t_local]),
                        'height': frame_H, 'width': frame_W,
                    }
                    proj_2d = project_tracks_3d_to_2d(
                        tracks3d=coords_3d[np.newaxis],
                        camera_views=[camera_view],
                    )[0]  # (N, 2) xy pixels
                    px = proj_2d[visible, 0]
                    py = proj_2d[visible, 1]
                    patch_x = np.clip((px / frame_W * 16).astype(int), 0, 15)
                    patch_y = np.clip((py / frame_H * 16).astype(int), 0, 15)
                    covered_patches[t_local, patch_y, patch_x] = True

            # Dilate: mark 8-neighbors of each covered patch as covered too
            from scipy.ndimage import binary_dilation
            struct = np.ones((1, 3, 3), dtype=bool)  # 3x3 per frame, no cross-frame
            covered_patches = binary_dilation(covered_patches, structure=struct)

            n_covered = covered_patches.sum()
            n_total = T_chunk * 16 * 16
            uncovered_frac = 1.0 - n_covered / n_total
            logger.info(
                f"  Covered patches: {n_covered}/{n_total} "
                f"({100 * n_covered / n_total:.1f}%), "
                f"uncovered fraction: {uncovered_frac:.2f}"
            )

        # Scale keypoints-per-cluster by uncovered fraction
        if covered_patches is not None:
            uncovered_frac = 1.0 - covered_patches.sum() / (T_chunk * 16 * 16)
            scaled_points = max(1, int(args.num_points_per_entity * uncovered_frac))
            scaled_merge_ratio = max(1, int(args.merge_ratio * uncovered_frac))
        else:
            scaled_points = args.num_points_per_entity
            scaled_merge_ratio = args.merge_ratio

        _log_vram(f"Chunk {chunk_idx} before DINO")
        _dino_t0 = time.time()
        logger.info(
            f"  Extracting DINO keypoints on chunk {chunk_idx} "
            f"(points_per_entity={scaled_points}, merge_ratio={scaled_merge_ratio})…"
        )
        chunk_peak_frame_data, _ = extract_dino_keypoints(
            chunk_video,
            dino_extractor,
            clustering_method=args.clustering_method,
            merge_ratio=scaled_merge_ratio,
            clustering_num_iters=args.clustering_num_iters,
            n_clusters=args.n_clusters,
            num_points_per_entity=scaled_points,
            use_connected_components=args.use_connected_components,
            debug=args.debug,
            covered_patches=covered_patches,
            dino_stride=args.dino_stride,
        )

        logger.info(f"  [TIMER] DINO extraction took {time.time() - _dino_t0:.1f}s")
        _log_vram(f"Chunk {chunk_idx} after DINO")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ==============================================================
        #  Step C: Track new keypoints through this chunk
        # ==============================================================
        # Move tracker to GPU, DINO to CPU
        _move_model_to(dino_extractor, "cpu")
        _move_model_to(model_3dtracker, args.device)

        new_peak_list = sorted(chunk_peak_frame_data.keys())

        # Batch all new keypoints across peak frames into a single tracking call
        batched_queries = []
        batched_slices = []  # (peak_global, cluster_ids, points_xy, start_idx, end_idx)
        offset = 0

        for peak_local in new_peak_list:
            peak_global = cs + peak_local
            pfd = chunk_peak_frame_data[peak_local]
            points_xy = pfd['points']
            cluster_ids = pfd['cluster_ids']
            n_pts = points_xy.shape[0]

            if n_pts == 0:
                continue

            # Build 2D query: [frame_local, x, y]
            frame_col = np.full((n_pts, 1), peak_local, dtype=np.float32)
            new_query = np.concatenate(
                [frame_col, points_xy.astype(np.float32)], axis=1
            )
            batched_queries.append(new_query)
            batched_slices.append((peak_global, cluster_ids, points_xy, offset, offset + n_pts))
            offset += n_pts

        if batched_queries:
            all_new_queries = np.concatenate(batched_queries, axis=0)
            logger.info(
                f"  Tracking {all_new_queries.shape[0]} new keypoints from "
                f"{len(batched_slices)} peak frames in one batched call"
            )
            _track_t0 = time.time()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            coords_all, visibs_all = _run_tracking_on_chunk(
                model_3dtracker,
                chunk_video, chunk_depth, chunk_intrs, chunk_extrs,
                [all_new_queries],
                frame_H, frame_W, args,
                queries_are_3d=False,
            )

            coords_all_np = coords_all.cpu().numpy() if torch.is_tensor(coords_all) else coords_all
            visibs_all_np = visibs_all.cpu().numpy() if torch.is_tensor(visibs_all) else visibs_all
            if coords_all_np.ndim == 2:
                coords_all_np = coords_all_np[:, np.newaxis, :]
            if visibs_all_np.ndim == 1:
                visibs_all_np = visibs_all_np[:, np.newaxis]

            logger.info(f"  [TIMER] Batched tracking took {time.time() - _track_t0:.1f}s")
            _log_vram(f"Chunk {chunk_idx} after batched tracking")

            # Split results back into per-group trajectories
            for peak_global, cluster_ids, points_xy, si, ei in batched_slices:
                coords_new_np = coords_all_np[:, si:ei, :]
                visibs_new_np = visibs_all_np[:, si:ei]

                trajectory_groups[peak_global] = {
                    'coords_chunks': [coords_new_np],
                    'visibs_chunks': [visibs_new_np],
                    'cluster_ids': cluster_ids,
                    'peak_frame_global': peak_global,
                    'first_global_frame': cs,
                    'frame_ranges': [(cs, ce)],
                }
                group_order.append(peak_global)
                query_points_per_frame[peak_global] = points_xy

            logger.info(
                f"  Registered {len(batched_slices)} groups from batched tracking"
            )

        # ==============================================================
        #  Step D: Update active groups
        # ==============================================================
        # All groups that have been registered so far are propagated to next chunk
        active_groups = list(trajectory_groups.keys())

        logger.info(
            f"  Chunk {chunk_idx} done: {len(trajectory_groups)} total groups, "
            f"{len(active_groups)} active for next chunk "
            f"({time.time() - _chunk_t0:.1f}s)"
        )
        _log_vram(f"Chunk {chunk_idx} end")

    logger.info(f"[TIMER] Progressive tracking (all chunks) took {time.time() - _stage4_t0:.1f}s")
    _log_vram("Before assembly")

    # ==================================================================
    #  5. Assemble global trajectories → query_frame_results format
    # ==================================================================
    _stage5_t0 = time.time()
    logger.info("Assembling global trajectories…")

    query_frame_results = {}
    cluster_ids_per_frame = {}
    is_moving_per_frame = {}
    max_disp_per_frame = {}

    for gi, gpf in enumerate(group_order):
        tg = trajectory_groups[gpf]

        # Concatenate coords and visibs across all chunks
        coords_full = np.concatenate(tg['coords_chunks'], axis=0)  # (T_life, N, 3)
        visibs_full = np.concatenate(tg['visibs_chunks'], axis=0)  # (T_life, N)

        # Build frame_indices
        frame_indices = []
        for fr_start, fr_end in tg['frame_ranges']:
            frame_indices.extend(range(fr_start, fr_end))

        T_life = coords_full.shape[0]
        assert T_life == len(frame_indices), (
            f"Group {gpf}: trajectory length {T_life} != "
            f"frame_indices length {len(frame_indices)}"
        )

        # Build intrinsics/extrinsics segments for this group's lifespan
        intrs_seg = intrs_npy[frame_indices]
        extrs_seg = extrs_w2c[frame_indices]

        # Convert to torch tensors — keep on CPU to avoid GPU accumulation
        # (compute_is_moving converts to numpy internally anyway)
        coords_tensor = torch.from_numpy(coords_full).float()
        visibs_tensor = torch.from_numpy(visibs_full).float()
        intrs_tensor = torch.from_numpy(intrs_seg).float()
        extrs_tensor = torch.from_numpy(extrs_seg).float()
        # NOTE: do NOT materialize a per-group depth slice here — duplicating
        # depth across all groups causes huge host-RAM usage (OOM on long
        # videos). Consumers that need depth use the full `depths` tensor
        # passed alongside `query_frame_results` and index by `frame_indices`.

        # ── Motion detection (runs on CPU/numpy) ──
        moving_thr = float(np.sqrt((frame_H / 20.0) ** 2 + (frame_W / 20.0) ** 2))
        is_moving, max_disp = compute_is_moving(
            coords_tensor, visibs_tensor,
            intrs_tensor, extrs_tensor,
            frame_H, frame_W,
            threshold=moving_thr,
        )

        cluster_ids = tg['cluster_ids']
        cluster_ids_per_frame[gpf] = cluster_ids
        is_moving_per_frame[gpf] = is_moving
        max_disp_per_frame[gpf] = max_disp

        n_moving = int(is_moving.sum())
        logger.info(
            f"Group {gpf}: {coords_full.shape[1]} keypoints, "
            f"{n_moving} moving / {coords_full.shape[1] - n_moving} static, "
            f"lifespan {len(frame_indices)} frames"
        )
        # Log VRAM every 10 groups
        if gi % 10 == 0:
            _log_vram(f"Assembly group {gi}/{len(group_order)}")

        query_frame_results[gpf] = {
            "coords": coords_tensor,
            "visibs": visibs_tensor,
            "video_segment": chunk_video,  # placeholder
            "intrinsics_segment": intrs_tensor,
            "extrinsics_segment": extrs_tensor,
            "frame_indices": frame_indices,
        }

    logger.info(f"[TIMER] Assembly took {time.time() - _stage5_t0:.1f}s")
    _log_vram("After assembly")

    # ==================================================================
    #  6. Build return dict
    # ==================================================================
    # Always return the FULL depth/intrinsics/extrinsics (not a per-group slice)
    depths_out = torch.from_numpy(depth_npy).float()
    intrinsics_out = torch.from_numpy(intrs_npy).float()
    extrinsics_out = torch.from_numpy(extrs_w2c).float()

    if query_frame_results:
        first_key = min(query_frame_results.keys())
        coords = query_frame_results[first_key]["coords"]
        visibs = query_frame_results[first_key]["visibs"]
    else:
        coords = torch.empty((0, 0, 3))
        visibs = torch.empty((0, 0))

    logger.info(f"[TIMER] Total process_long_video took {time.time() - _pipeline_t0:.1f}s")
    _log_vram("Pipeline end")

    return {
        "video_tensor": video_ten[:tracking_chunk_size],  # first chunk for compat
        "video_tensor_full": video_ten,
        "video_tensor_orig": video_tensor,  # pre-VGGT original resolution
        "depths": depths_out,
        "coords": coords,
        "visibs": visibs,
        "intrinsics": intrinsics_out,
        "extrinsics": extrinsics_out,
        "query_points_per_frame": query_points_per_frame,
        "original_filenames": original_filenames,
        "depth_conf": depth_conf_npy,
        "query_frame_results": query_frame_results,
        "full_intrinsics": torch.from_numpy(intrs_npy).float().to(args.device),
        "full_extrinsics": torch.from_numpy(extrs_w2c).float().to(args.device),
        "cluster_ids_per_frame": cluster_ids_per_frame,
        "is_moving_per_frame": is_moving_per_frame,
        "max_disp_per_frame": max_disp_per_frame,
    }


def find_video_folders(base_path: str, scan_depth: int = 2):
    """
    Recursively scan subfolders up to a given depth and return inputs
    that contain images (.jpg/.jpeg/.png) or stand-alone video files
    (.mp4/.webm/etc.).

    Args:
        base_path: Root directory to scan
        scan_depth: Number of directory levels to traverse

    Returns:
        List of folder paths containing image files at the target depth
    """
    img_exts = (".jpg", ".jpeg", ".png")
    video_exts = VIDEO_EXTS

    # Normalize the base path
    base_path = os.path.abspath(base_path.rstrip(os.sep))
    base_depth = base_path.count(os.sep)
    target_depth = base_depth + scan_depth

    video_folders = []

    for root, dirs, files in os.walk(base_path):
        current_depth = os.path.abspath(root.rstrip(os.sep)).count(os.sep)

        # Skip folders above the target depth
        if current_depth < target_depth:
            continue

        # Select only folders/files exactly at the target depth
        if current_depth == target_depth:
            has_images = any(f.lower().endswith(img_exts) for f in files)
            if has_images:
                video_folders.append(root)
            # Also collect individual video files at this depth
            for f in files:
                if f.lower().endswith(video_exts):
                    video_folders.append(os.path.join(root, f))

        # Skip deeper folders for performance (no need to go further)
        if current_depth > target_depth:
            dirs[:] = []  # prevent os.walk from descending further

    # Deduplicate and sort for stable ordering
    video_folders = sorted(list(dict.fromkeys(video_folders)))
    return video_folders


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg")
_NAT_SPLIT = re.compile(r"(\d+)")


def _natural_key(path):
    """Sort key that orders embedded integers numerically (foo_2 < foo_10)."""
    parts = _NAT_SPLIT.split(os.path.basename(path))
    return [int(p) if p.isdigit() else p for p in parts]


def resize_frames(video_tensor, target_hw, is_depth=False):
    """Resize (N, C, H, W) RGB or (N, H, W) depth frames to ``target_hw``.

    Bilinear for RGB, nearest for depth. No-op if ``target_hw`` is None or
    the frames are already at the target size.
    """
    if target_hw is None:
        return video_tensor
    tgt_h, tgt_w = int(target_hw[0]), int(target_hw[1])
    cur_h, cur_w = int(video_tensor.shape[-2]), int(video_tensor.shape[-1])
    if (cur_h, cur_w) == (tgt_h, tgt_w):
        return video_tensor
    squeeze_after = video_tensor.dim() == 3  # (N, H, W) — depth grayscale
    if squeeze_after:
        video_tensor = video_tensor.unsqueeze(1)
    if is_depth:
        video_tensor = torch.nn.functional.interpolate(
            video_tensor, size=(tgt_h, tgt_w), mode="nearest",
        )
    else:
        video_tensor = torch.nn.functional.interpolate(
            video_tensor, size=(tgt_h, tgt_w),
            mode="bilinear", align_corners=False,
        )
    if squeeze_after:
        video_tensor = video_tensor.squeeze(1)
    logger.info(f"Resized video from ({cur_h},{cur_w}) to ({tgt_h},{tgt_w})")
    return video_tensor


def load_video_and_mask(video_path, mask_dir=None, frame_step=1, is_depth=False,
                        frame_range=None, target_fps=0, target_hw=None):
    """Load video frames with uniform frame stepping.

    Args:
        frame_step: take every Nth frame for image inputs (1 = no skip).
        frame_range: optional (start, end) tuple to load only a slice of the
            source frames before stepping.
        target_fps: target FPS for video file inputs. If the video's native FPS
            exceeds this value, frames are subsampled to approximate this rate.
            Set to 0 to fall back to frame_step for video inputs as well.
        target_hw: optional (H, W) to resize all frames to after loading.
            Bilinear for RGB, nearest for depth. Skipped if the video is
            already at the target size.
    """
    original_filenames = []

    if os.path.isdir(video_path):
        # --- Image directory: always use frame_step ---
        img_files = [
            os.path.join(video_path, f) for f in os.listdir(video_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        img_files = sorted(img_files, key=_natural_key)

        # Slice to chunk range
        if frame_range is not None:
            img_files = img_files[frame_range[0]:frame_range[1]]

        # Frame stepping
        if frame_step > 1:
            img_files = img_files[::frame_step]

        video_tensor = []
        for img_file in tqdm.tqdm(img_files, desc="Loading images"):
            img = Image.open(img_file)
            if is_depth:
                img = img.convert("I;16")  # 16-bit grayscale for depth
            else:
                img = img.convert("RGB")
            video_tensor.append(
                torch.from_numpy(np.array(img)).float()
            )
            filename = os.path.splitext(os.path.basename(img_file))[0]
            original_filenames.append(filename)
        video_tensor = torch.stack(video_tensor)  # (N, H, W, 3)
    elif video_path.lower().endswith(VIDEO_EXTS):
        video_array = media.read_video(video_path)
        source_fps = getattr(getattr(video_array, 'metadata', None), 'fps', None)
        # mediapy occasionally fails to decode otherwise-valid mp4s and returns
        # an empty (0,)-shaped array. Fall back to imageio in that case.
        if np.asarray(video_array).ndim != 4:
            logger.warning(
                f"mediapy returned shape {np.asarray(video_array).shape} for "
                f"{video_path}; falling back to imageio"
            )
            import imageio.v3 as iio
            video_array = iio.imread(video_path, plugin="pyav")
        video_tensor = torch.from_numpy(np.asarray(video_array))
        for i in range(len(video_tensor)):
            original_filenames.append(f"frame_{i:010d}")

        # Slice to chunk range
        if frame_range is not None:
            video_tensor = video_tensor[frame_range[0]:frame_range[1]]
            original_filenames = original_filenames[frame_range[0]:frame_range[1]]

        # Determine effective frame_step for video inputs
        effective_step = frame_step
        if target_fps > 0:
            if source_fps is not None and source_fps > target_fps:
                effective_step = max(1, round(source_fps / target_fps))
                logger.info(
                    f"Video FPS={source_fps:.1f}, target_fps={target_fps:.1f} "
                    f"→ effective frame_step={effective_step}"
                )
            elif source_fps is not None:
                effective_step = 1  # already sparse enough
                logger.info(
                    f"Video FPS={source_fps:.1f} <= target_fps={target_fps:.1f} "
                    f"→ no subsampling"
                )
            else:
                logger.warning(
                    "Could not read video FPS metadata; falling back to "
                    f"frame_step={frame_step}"
                )

        if effective_step > 1:
            video_tensor = video_tensor[::effective_step]
            original_filenames = original_filenames[::effective_step]
    else:
        raise ValueError(
            f"Unsupported input {video_path!r}: expected a folder of .jpg/.png "
            f"frames or a video file ({', '.join(VIDEO_EXTS)})"
        )

    if not is_depth:
        video_tensor = video_tensor.permute(
            0, 3, 1, 2
        )  # Convert to tensor and permute to (N, C, H, W)
    video_tensor = video_tensor.float()

    video_tensor = resize_frames(video_tensor, target_hw, is_depth=is_depth)

    video_length = len(video_tensor)
    logger.debug(f"Loaded video with {video_length} frames from {video_path}")
    frame_h, frame_w = video_tensor.shape[-2:]

    video_mask_npy = None
    if mask_dir is not None:
        video_mask_npy = []
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

        for mask_file in mask_files:
            mask = media.read_image(mask_file)
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
            video_mask_npy.append(mask)
        video_mask_npy = np.stack(video_mask_npy)

    if not is_depth:
        video_tensor /= 255.
    return video_tensor, video_mask_npy, original_filenames


def prepare_query_points(query_xyt, depths, intrinsics, extrinsics):
    """Back-project 2D (t, x, y) queries to 3D (t, wx, wy, wz) world coords.

    Rows within a single ``query_i`` array may reference different frame
    indices — each row's depth/intrinsics/extrinsics are sampled using its
    own ``t``. (The earlier implementation took ``t`` from row 0 only, so
    batching queries from multiple peak frames into one array anchored every
    point to the first row's frame and silently corrupted tracks.)
    """
    final_queries = []
    for query_i in query_xyt:
        if len(query_i) == 0:
            continue

        t_col = query_i[:, 0].astype(np.int64)           # (N,)
        xy = query_i[:, 1:3].astype(np.float32)           # (N, 2)
        ji = np.round(xy).astype(np.int64)                # (N, 2)  pixel (x, y)
        H_d, W_d = depths.shape[-2], depths.shape[-1]
        ji[:, 0] = np.clip(ji[:, 0], 0, W_d - 1)
        ji[:, 1] = np.clip(ji[:, 1], 0, H_d - 1)

        d = depths[t_col, ji[:, 1], ji[:, 0]]             # (N,) depth per row's t

        # Vectorized per-row unprojection: x_world = c2w_t · (d * K_t^-1 · [x, y, 1])
        K_inv = np.linalg.inv(intrinsics[t_col])          # (N, 3, 3)
        c2w = np.linalg.inv(extrinsics[t_col])            # (N, 4, 4)

        ones = np.ones_like(xy[:, :1])
        xy_homo = np.concatenate([xy, ones], axis=-1)[..., None]   # (N, 3, 1)
        local_coords = (K_inv @ xy_homo).squeeze(-1) * d[:, None]  # (N, 3)
        world_coords = (c2w[:, :3, :3] @ local_coords[..., None]).squeeze(-1) \
                       + c2w[:, :3, 3]                             # (N, 3)

        final_queries.append(
            np.concatenate([query_i[:, :1], world_coords], axis=-1)
        )
    return np.concatenate(final_queries, axis=0)  # (N, 4)


def prepare_inputs(
    video_ten,
    depths,
    intrinsics,
    extrinsics,
    query_point,
    inference_res: Tuple[int, int],
    support_grid_size: int,
    num_threads: int = 8,
    device: str = "cuda",
    queries_are_3d: bool = False,
):
    _original_res = depths.shape[1:3]
    inference_res = _original_res  # fix as the same

    intrinsics[:, 0, :] *= (inference_res[1] - 1) / (_original_res[1] - 1)
    intrinsics[:, 1, :] *= (inference_res[0] - 1) / (_original_res[0] - 1)

    # resize & remove edges
    with ThreadPoolExecutor(num_threads) as executor:
        depths_futures = [
            executor.submit(_filter_one_depth, depth, 0.08, 15, intrinsic)
            for depth, intrinsic in zip(depths, intrinsics)
        ]
        depths = np.stack([future.result() for future in depths_futures])

    if queries_are_3d:
        # query_point is already [t, world_x, world_y, world_z] — skip unprojection
        if isinstance(query_point, list):
            query_point = np.concatenate(query_point, axis=0)
        query_point = torch.from_numpy(query_point).float().to(device)
    else:
        query_point = prepare_query_points(query_point, depths, intrinsics, extrinsics)
        query_point = torch.from_numpy(query_point).float().to(device)

    video = (video_ten.float()).to(device).clamp(0, 1)
    depths = torch.from_numpy(depths).float().to(device)
    intrinsics = torch.from_numpy(intrinsics).float().to(device)
    extrinsics = torch.from_numpy(extrinsics).float().to(device)

    return video, depths, intrinsics, extrinsics, query_point, support_grid_size


LOCK_TIMEOUT_S = 1.5 * 3600  # treat older locks as stale; raise if one episode takes longer


def _output_complete(output_path):
    """True if an episode directory has its final (non-empty) samples/ arrays."""
    required = [
        os.path.join(output_path, "samples", name)
        for name in ("frame_indices.npy", "keypoints.npy", "traj.npy")
    ]
    return all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in required)


def _lock_owner_is_dead(lock_path):
    """True if the lock was written by a process on this host that no longer exists."""
    try:
        with open(lock_path) as f:
            info = dict(kv.split("=", 1) for kv in f.read().split())
        if info.get("host") != socket.gethostname():
            return False
        os.kill(int(info["pid"]), 0)
    except ProcessLookupError:
        return True
    except (OSError, ValueError, KeyError, OverflowError):
        return False
    return False


def _acquire_lock(lock_path):
    """Atomically create ``lock_path``; reclaim it if stale. Returns True if acquired.

    A lock is stale when older than LOCK_TIMEOUT_S or when its owner process
    (on this host) is gone, e.g. after an OOM kill. Only the worker holding the
    short-lived ``<lock>.reclaim`` lock may replace a stale lock, so when
    several workers race for the same stale lock exactly one wins.
    """
    lock_info = f"pid={os.getpid()} host={socket.gethostname()} t={time.time()}\n"

    def create(path, info):
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, info.encode())
        os.close(fd)

    def read(path):
        with open(path) as f:
            return f.read()

    try:
        create(lock_path, lock_info)
        return True
    except FileExistsError:
        pass
    try:
        stale_info = read(lock_path)
        if (time.time() - os.path.getmtime(lock_path) <= LOCK_TIMEOUT_S
                and not _lock_owner_is_dead(lock_path)):
            return False
        reclaim_path = lock_path + ".reclaim"
        try:
            create(reclaim_path, lock_info)
        except FileExistsError:
            # Someone else is reclaiming; clear a reclaim lock left by a crash.
            if time.time() - os.path.getmtime(reclaim_path) > 60:
                os.remove(reclaim_path)
            return False
        try:
            if read(lock_path) != stale_info:
                return False  # already replaced by another worker
            os.remove(lock_path)
            logger.warning(f"Removed stale lock {lock_path}")
            create(lock_path, lock_info)
            return True
        finally:
            try:
                os.remove(reclaim_path)
            except FileNotFoundError:
                pass
    except OSError:
        return False


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Save logs to file in the output directory
    node_name = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    log_path = os.path.join(out_dir, f"{node_name}_infer.log")
    logger.add(log_path, mode="a", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}")
    logger.info(f"Logging to {log_path}")

    # initialize 3D models
    model_depth_pose = video_depth_pose_dict[args.depth_pose_method](args)
    model_3dtracker = load_model(args.checkpoint).to(args.device)

    # initialize DINO feature extractor for keypoint extraction
    logger.info("Loading DINO feature extractor …")
    dino_extractor = create_dino_extractor()
    logger.info("DINO feature extractor loaded")

    # Determine video paths to process
    dataset = get_dataset(args.dataset_name) if args.dataset_name else None
    if dataset is not None:
        video_folders = dataset.find_episodes(args)
        depth_folders = [None] * len(video_folders)
        logger.info(f"Found {len(video_folders)} {dataset.name} episodes to process")
        if not video_folders:
            logger.error(f"No {dataset.name} episodes found in {args.video_path}")
            exit(1)
    elif args.batch_process:
        video_folders = find_video_folders(args.video_path, args.scan_depth)
        if args.depth_path is not None:
            depth_folders = find_video_folders(args.depth_path)
            if len(depth_folders) != len(video_folders):
                logger.error(
                    f"Number of depth folders ({len(depth_folders)}) does not "
                    f"match number of video folders ({len(video_folders)})"
                )
                exit(1)
        else:
            depth_folders = [None] * len(video_folders)

        logger.info(f"Found {len(video_folders)} video folders to process")
        if not video_folders:
            logger.error(f"No video folders found in {args.video_path}")
            exit(1)
    else:
        video_folders = [args.video_path]
        depth_folders = [args.depth_path]

    def episode_name(path):
        if dataset is not None:
            return dataset.episode_name(path)
        return os.path.basename(path.rstrip("/"))

    # Shuffle so multiple workers don't race on the same videos in order
    pairs = list(zip(video_folders, depth_folders))
    if args.episode_list:
        with open(args.episode_list) as f:
            keep = {line.strip() for line in f if line.strip()}
        pairs = [p for p in pairs if episode_name(p[0]) in keep]
        logger.info(f"--episode_list: kept {len(pairs)} episodes ({len(keep)} names listed)")
        if not pairs:
            logger.error(f"No episodes under {args.video_path} match {args.episode_list}")
            exit(1)
    random.shuffle(pairs)
    if args.max_episodes is not None and args.max_episodes > 0:
        pairs = pairs[:args.max_episodes]
        logger.info(f"--max_episodes={args.max_episodes}: truncated work list to {len(pairs)} pairs")

    # Process each video with the long-video pipeline
    summary = {"processed": 0, "skipped_existing": 0, "skipped_locked": 0,
               "skipped_low_coverage": 0}
    failed = []
    for video_path, depth_path in pairs:
        video_name = episode_name(video_path)

        output_path = os.path.join(out_dir, video_name)
        low_cov_path = os.path.join(out_dir, f".low_coverage_{video_name}")
        lock_path = os.path.join(out_dir, f".lock_{video_name}")
        lock_acquired = False
        if args.skip_existing:
            # Episodes already rejected for low coverage at this threshold.
            if os.path.isfile(low_cov_path):
                try:
                    with open(low_cov_path) as f:
                        prev_cov = float(f.read().strip())
                except (OSError, ValueError):
                    prev_cov = None
                if prev_cov is not None and prev_cov < args.min_track_coverage:
                    logger.info(
                        f"Skipping {video_name} - previously rejected for low track "
                        f"coverage ({prev_cov:.2%}); delete {low_cov_path} to retry"
                    )
                    summary["skipped_low_coverage"] += 1
                    continue
            if _output_complete(output_path):
                logger.info(f"Skipping {video_name} - output already exists")
                summary["skipped_existing"] += 1
                continue
            # Per-episode lock so several workers can share one out_dir.
            lock_acquired = _acquire_lock(lock_path)
            if not lock_acquired:
                logger.info(f"Skipping {video_name} - locked by another process")
                summary["skipped_locked"] += 1
                continue

        try:
            if args.skip_existing:
                # Re-check now that we hold the lock: another worker may have just
                # finished it. Otherwise wipe any partial output from a dead run.
                if _output_complete(output_path):
                    logger.info(f"Skipping {video_name} - output already exists")
                    summary["skipped_existing"] += 1
                    continue
                if os.path.exists(output_path):
                    logger.warning(
                        f"Detected incomplete output for {video_name} "
                        f"(missing/empty samples/{{frame_indices,keypoints,traj}}.npy); "
                        f"removing {output_path} and re-processing"
                    )
                    shutil.rmtree(output_path, ignore_errors=True)

            # Clear CUDA cache before processing each video
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            episode_meta = None
            if dataset is not None and dataset.read_meta is not None:
                episode_meta = dataset.read_meta(video_path)

            result = process_long_video(
                video_path, depth_path, args,
                model_3dtracker, model_depth_pose, dino_extractor,
            )

            # ── Track coverage quality check ──
            if result.get("query_frame_results"):
                _cov_t0 = time.time()
                per_frame_cov, mean_cov = compute_track_coverage(
                    result["query_frame_results"],
                    result["full_intrinsics"],
                    result["full_extrinsics"],
                    height=result["video_tensor"].shape[-2],
                    width=result["video_tensor"].shape[-1],
                )
                logger.info(
                    f"Track coverage: mean={mean_cov:.2%}, "
                    f"min={per_frame_cov.min():.2%}, max={per_frame_cov.max():.2%} "
                    f"({time.time() - _cov_t0:.1f}s)"
                )
                if mean_cov < args.min_track_coverage:
                    logger.warning(
                        f"Skipping {video_name}: track coverage {mean_cov:.2%} "
                        f"< threshold {args.min_track_coverage:.0%}. "
                        f"Video likely has excessive camera rotation."
                    )
                    with open(low_cov_path, "w") as f:
                        f.write(f"{mean_cov}\n")
                    summary["skipped_low_coverage"] += 1
                    continue

            # Save structured data
            save_structured_data(
                video_name=video_name,
                output_dir=out_dir,
                video_tensor=result["video_tensor"],
                depths=result["depths"],
                coords=result["coords"],
                visibs=result["visibs"],
                intrinsics=result["intrinsics"],
                extrinsics=result["extrinsics"],
                query_points_per_frame=result["query_points_per_frame"],
                original_filenames=result["original_filenames"],
                query_frame_results=result.get("query_frame_results"),
                future_len=args.future_len,
                history_len=args.history_len,
                cluster_ids_per_frame=result.get("cluster_ids_per_frame"),
                is_moving_per_frame=result.get("is_moving_per_frame"),
                max_disp_per_frame=result.get("max_disp_per_frame"),
                video_tensor_full=result.get("video_tensor_full"),
                video_tensor_orig=result.get("video_tensor_orig"),
            )

            if episode_meta is not None:
                description, meta = episode_meta
                ep_dir = os.path.join(out_dir, video_name)
                os.makedirs(ep_dir, exist_ok=True)
                if description:
                    with open(os.path.join(ep_dir, "description.txt"), "w") as f:
                        f.write(description + "\n")
                with open(os.path.join(ep_dir, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

            # ── Debug video ──
            if args.debug and result.get("query_frame_results"):
                debug_path = os.path.join(
                    out_dir, video_name, f"{video_name}_debug.mp4")
                debug_H, debug_W = result["video_tensor_full"].shape[-2:]
                moving_thr_debug = float(np.sqrt((debug_H / 20.0) ** 2 + (debug_W / 20.0) ** 2))
                save_debug_video(
                    video_tensor=result["video_tensor_full"],
                    query_frame_results=result["query_frame_results"],
                    is_moving_per_frame=result["is_moving_per_frame"],
                    cluster_ids_per_frame=result["cluster_ids_per_frame"],
                    output_path=debug_path,
                    motion_length_threshold=moving_thr_debug,
                )

                debug_fixed_path = os.path.join(
                    out_dir, video_name, f"{video_name}_debug_fixed_pov.mp4")
                save_debug_video_fixed_pov(
                    video_tensor=result["video_tensor_full"],
                    query_frame_results=result["query_frame_results"],
                    cluster_ids_per_frame=result["cluster_ids_per_frame"],
                    output_path=debug_fixed_path,
                    is_moving_per_frame=result.get("is_moving_per_frame"),
                    chunk_size=getattr(args, 'chunk_size', None),
                )

            # Save the legacy per-video bundle only if explicitly requested.
            # Downstream tools now read cameras.npz + samples/<name>_<t>.npz.
            if args.save_main_npz:
                video_dir = os.path.join(out_dir, video_name)
                os.makedirs(video_dir, exist_ok=True)
                data_npz_load = {}
                data_npz_load["coords"] = result["coords"].cpu().numpy()
                data_npz_load["extrinsics"] = result["full_extrinsics"].cpu().numpy()
                data_npz_load["intrinsics"] = result["full_intrinsics"].cpu().numpy()
                data_npz_load["height"] = result["video_tensor"].shape[-2]
                data_npz_load["width"] = result["video_tensor"].shape[-1]
                data_npz_load["depths"] = result["depths"].cpu().numpy().astype(np.float16)
                data_npz_load["unc_metric"] = result["depth_conf"].astype(np.float16)
                data_npz_load["visibs"] = result["visibs"][..., None].cpu().numpy()
                if args.save_video:
                    data_npz_load["video"] = result["video_tensor"].cpu().numpy()

                save_path = os.path.join(video_dir, video_name + ".npz")
                np.savez(save_path, **data_npz_load)
                logger.info(f"Legacy per-video NPZ saved to {save_path}")

            summary["processed"] += 1
            if os.path.isfile(low_cov_path):
                os.remove(low_cov_path)

        except Exception as e:
            import traceback

            logger.error(f"Failed to process {video_name}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            failed.append(video_name)
        finally:
            # Release the lock on every exit path (success, low-coverage skip,
            # failure) so other workers / later runs are not blocked.
            if lock_acquired:
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass

    # Cleanup
    del model_3dtracker
    del model_depth_pose
    del dino_extractor
    torch.cuda.empty_cache()
    logger.info(
        "Batch processing completed: "
        + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in summary.items())
        + f", {len(failed)} failed"
    )
    if failed:
        logger.warning(f"Failed episodes (see errors above): {', '.join(sorted(failed))}")
        exit(1)
