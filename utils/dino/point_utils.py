"""Utilities for DINO-based keypoint extraction: peak frames, connected
components, and point-from-cluster sampling."""

import torch
import numpy as np
from PIL import Image
from collections import defaultdict


def get_cluster_peak_frames(cluster_labels, num_clusters=None, covered_patches=None):
    """
    Find the frame where each cluster has its maximum presence.

    Args:
        cluster_labels: numpy array of shape [T, H, W] containing cluster labels
        num_clusters: optional, maximum number of clusters to consider
        covered_patches: optional (T, H, W) bool array. True = patch is already
            tracked and should be excluded when counting cluster presence.

    Returns:
        peak_frames: dictionary mapping frame_id to list of cluster_ids.
    """
    T, H, W = cluster_labels.shape

    if num_clusters is None:
        num_clusters = np.max(cluster_labels) + 1

    peak_frames = {}

    for cluster_id in range(num_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        if covered_patches is not None:
            cluster_mask = cluster_mask & ~covered_patches
        cluster_counts = np.sum(cluster_mask, axis=(1, 2))

        if np.any(cluster_counts > 0):
            peak_frame = np.argmax(cluster_counts)
            peak_frames[cluster_id] = {
                'frame_id': peak_frame,
                'count': cluster_counts[peak_frame]
            }
    peak_frames_dict = defaultdict(list)
    for cluster_id, peak_frame_dict in peak_frames.items():
        peak_frames_dict[peak_frame_dict['frame_id']].append(cluster_id)

    return peak_frames_dict


def get_cluster_proportional_allocation(cluster_labels, num_clusters=None,
                                        covered_patches=None,
                                        num_points_per_entity=64):
    """Allocate keypoints per cluster proportionally across all frames where
    the cluster is present.

    Args:
        cluster_labels: ``(T, H, W)`` int array of cluster labels.
        num_clusters: max cluster id + 1 (auto-detected if *None*).
        covered_patches: optional ``(T, H, W)`` bool array.  ``True`` = patch
            already tracked (excluded from counting).
        num_points_per_entity: target keypoints per cluster (e.g. 64).

    Returns:
        allocation: ``dict[int, dict[int, int]]`` —
            ``allocation[cluster_id][frame_id] = n_keypoints``.
    """
    T, H, W = cluster_labels.shape

    if num_clusters is None:
        num_clusters = int(np.max(cluster_labels)) + 1

    allocation = {}

    for cluster_id in range(num_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        if covered_patches is not None:
            cluster_mask = cluster_mask & ~covered_patches
        counts = np.sum(cluster_mask, axis=(1, 2))  # (T,)

        total = int(counts.sum())
        if total == 0:
            continue

        frame_alloc = {}
        for t in range(T):
            c = int(counts[t])
            if c == 0:
                continue
            raw = num_points_per_entity * c / total
            n = round(raw)
            if n < 1:
                continue
            frame_alloc[t] = n

        if frame_alloc:
            allocation[cluster_id] = frame_alloc

    return allocation


def find_connected_components(mask, connectivity=4):
    """Find connected components in a binary mask.

    Args:
        mask: Binary mask of shape (H, W)
        connectivity: 4 or 8 for connectivity type

    Returns:
        components: List of arrays containing (y,x) coordinates for each component
    """
    from scipy.ndimage import label

    if connectivity == 4:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])
    else:
        structure = np.ones((3, 3))

    labeled_array, num_features = label(mask, structure=structure)

    components = []
    for i in range(1, num_features + 1):
        y_indices, x_indices = np.where(labeled_array == i)
        component_coords = np.stack([y_indices, x_indices], axis=1)
        components.append(component_coords)

    return components


def create_overlay_mask(image, labels):
    """Create a colored overlay mask based on clustering labels."""
    colors = [
        (255, 205, 0), (0, 200, 124), (0, 92, 180), (226, 26, 91),
        (150, 111, 51), (255, 110, 36), (124, 0, 160), (128, 128, 128),
        (255, 0, 127), (0, 180, 216), (144, 238, 144), (255, 69, 0),
        (147, 112, 219), (0, 163, 108), (255, 174, 66), (106, 90, 205),
        (250, 128, 114), (72, 209, 204), (255, 218, 185), (153, 50, 204),
        (0, 139, 139), (255, 99, 71), (186, 85, 211), (60, 179, 113),
        (221, 160, 221), (100, 149, 237), (219, 112, 147), (176, 196, 222),
        (255, 127, 80), (102, 205, 170), (238, 130, 238), (64, 224, 208),
    ]

    h, w = labels.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(colors)):
        mask[labels == i] = colors[i]

    mask = Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)
    return Image.blend(image, mask, 0.5)
