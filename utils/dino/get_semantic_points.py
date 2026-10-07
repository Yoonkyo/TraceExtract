"""
Functions to extract semantic keypoints from DINO cluster labels.
"""
import os
import warnings
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from .clustering import (get_temporal_bipartite_clusters,
                         cluster_coordinates,
                         cluster_coordinates_per_component)
from .point_utils import get_cluster_peak_frames, get_cluster_proportional_allocation, find_connected_components
from .point_utils import create_overlay_mask


def make_frame_cluster_vis(video_frames, frame_id, feat_cluster_labels,
                           debug_vis_root):
    """Make a visualisation of the cluster points for a given frame."""
    original_frame = Image.fromarray(video_frames[0, frame_id])

    overlay_dino_global_torch = create_overlay_mask(original_frame,
                                                    feat_cluster_labels[frame_id])

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.imshow(video_frames[0, frame_id])
    ax1.set_title(f'Frame {frame_id}', fontsize=14)
    ax1.axis('off')

    ax2.imshow(overlay_dino_global_torch)
    ax2.set_title('DINO Global Cluster', fontsize=14)
    ax2.axis('off')

    plt.tight_layout()
    os.makedirs(debug_vis_root, exist_ok=True)
    plt.savefig(os.path.join(debug_vis_root, f"debug_cluster_pts_{frame_id}.png"),
                bbox_inches='tight', dpi=150)
    plt.close()


def get_points_in_cluster(args, labels, image_size, clusters_to_consider=None,
                          covered_patches=None, points_per_cluster=None):
    """Per cluster sample points such that the points are uniformly spread
    across the cluster.

    Args:
        args: Arguments (needs num_points_per_entity, use_connected_components)
        labels: 2d cluster labels for one frame
        image_size: (height, width)
        clusters_to_consider: cluster ids to consider
        covered_patches: optional (H_patch, W_patch) bool array. True = patch
            is already tracked and should be excluded from sampling.
        points_per_cluster: optional dict mapping cluster_id -> number of
            keypoints to sample for this frame.  Falls back to
            ``args.num_points_per_entity`` when *None* or when a cluster is
            missing from the dict.

    Returns:
        points, point_labels, component_labels
    """
    img_height, img_width = image_size

    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    points_list = []
    labels_list = []
    component_labels_list = []
    for label in clusters_to_consider:
        n_points = (points_per_cluster or {}).get(label, args.num_points_per_entity)

        mask = labels == label

        # Zero out patches that are already covered by existing tracks
        if covered_patches is not None:
            covered_t = torch.from_numpy(covered_patches).to(mask.device) if isinstance(
                covered_patches, np.ndarray) else covered_patches.to(mask.device)
            mask = mask & ~covered_t

        if not mask.any():
            continue  # all patches for this cluster are covered

        mask = torch.nn.functional.interpolate(
            mask[None, None, :, :].float(),
            size=(img_height, img_width),
            mode='nearest'
        )[0, 0]

        y_indices, x_indices = torch.where(mask)
        if args.use_connected_components:
            components = find_connected_components(mask.cpu().numpy(),
                                                   connectivity=4)
            if not components:
                warnings.warn(f"No connected components found for label {label}")
                continue
            cluster_points, component_labels = cluster_coordinates_per_component(
                                    components,
                                    n_points,
                                    cluster_coordinates_fn=cluster_coordinates)

            component_labels_list.extend(component_labels.tolist())

        else:
            stacked_indices = torch.stack([y_indices, x_indices], dim=1).cpu().numpy()
            cluster_points = cluster_coordinates(stacked_indices,
                                                 n_points)

        scaled_x = cluster_points[:, 1:2]
        scaled_y = cluster_points[:, 0:1]

        scaled_x = np.clip(scaled_x, 0, img_width - 1)
        scaled_y = np.clip(scaled_y, 0, img_height - 1)

        points_list.extend(np.concatenate([scaled_x, scaled_y], axis=1).tolist())
        labels_list.extend([label] * len(scaled_x))

    points = np.array(points_list)
    point_labels = np.array(labels_list)
    component_labels = np.array(component_labels_list)

    return points, point_labels, component_labels


def _get_cluster_patch_coords(cluster_label, feat_labels_frame,
                              covered_patches_frame):
    """Return (M, 2) patch-level (y, x) indices for one cluster in one frame,
    excluding covered patches.

    Operates directly on the 16x16 patch grid — no interpolate upscaling.
    Returns ``None`` if no valid patches remain.
    """
    if isinstance(feat_labels_frame, torch.Tensor):
        feat_labels_frame = feat_labels_frame.cpu().numpy()

    mask = (feat_labels_frame == cluster_label)

    if covered_patches_frame is not None:
        if isinstance(covered_patches_frame, torch.Tensor):
            covered_patches_frame = covered_patches_frame.cpu().numpy()
        mask = mask & ~covered_patches_frame

    if not mask.any():
        return None

    py, px = np.where(mask)
    return np.stack([py, px], axis=1).astype(np.int64)  # (M, 2) patch y,x


def get_points_from_clustering(args, video_frames, feat_extractor, debug_vis_root,
                               covered_patches=None):
    """Get points from clustering using proportional keypoint allocation.

    For each cluster, candidate pixels are **pooled across all allocated
    frames** and k-means is run once on the pool so that keypoints are
    spatially well-distributed even across frames.  Each selected point
    retains its source frame for tracking.

    Args:
        args: Arguments (needs ``num_points_per_entity``,
            ``use_connected_components``, etc.)
        video_frames: (1, T, H, W, C) numpy uint8
        feat_extractor: DINO feature extractor
        debug_vis_root: debug vis path
        covered_patches: optional (T, 16, 16) bool array. True = patch is
            already tracked. Excluded from counting and sampling.

    Returns:
        points_list, point_labels_list, component_labels_list, cluster_frame_id_list
    """
    _, n_frames, h, w, _ = video_frames.shape
    # extracting dino features
    dino_features = feat_extractor(video_frames, model_type='dino')[0]  # (T, 16, 16, 768); keep T even when T == 1

    # Apply clustering for all frame features
    if args.clustering_method == 'bipartite':
        feat_cluster_labels = get_temporal_bipartite_clusters(
                                        dino_features, merge_ratio=args.merge_ratio,
                                        num_iters=args.num_iters)
    elif args.clustering_method == 'kmeans':
        feat_cluster_labels, _ = feat_extractor.cluster_features(
                            dino_features, method='kmeans',
                            n_clusters=args.n_clusters, global_clustering=True,
                            use_torch=True
                        )
    else:
        raise ValueError(f"Clustering method not implemented {args.clustering_method}")

    # --- Proportional allocation: cluster_id -> {frame_id: n_keypoints} ---
    allocation = get_cluster_proportional_allocation(
        feat_cluster_labels,
        covered_patches=covered_patches,
        num_points_per_entity=args.num_points_per_entity,
    )

    # --- Cross-frame pooled sampling per cluster (patch-level k-means) ---
    # For each cluster: pool candidate *patch* coords from all allocated
    # frames, run k-means once on the 16x16 patch grid (~hundreds of
    # entries, vs ~100k+ pixel entries), assign selections back to source
    # frames via cKDTree, then jitter patch coords -> pixel coords so points
    # don't land on patch-center lattice.
    from collections import defaultdict
    from scipy.spatial import cKDTree

    # Patch cell size in pixels (16x16 patch grid over the full image)
    cell_h = h / 16.0
    cell_w = w / 16.0
    rng = np.random.default_rng(42)  # deterministic jitter

    # Collect results keyed by frame_id
    frame_points = defaultdict(list)       # frame_id -> list of (x, y)
    frame_labels = defaultdict(list)       # frame_id -> list of cluster_id
    frame_comp_labels = defaultdict(list)  # frame_id -> list of component_id

    for cluster_id, frame_alloc in allocation.items():
        # Total keypoints to sample for this cluster across all frames
        total_k = sum(frame_alloc.values())

        # Pool candidate patch coords: (M, 2) in patch y,x with a parallel
        # frame_id array
        all_patch_coords = []  # list of (M_i, 2) arrays
        all_frame_ids = []     # list of (M_i,) arrays

        for frame_id in sorted(frame_alloc.keys()):
            feat_labels_frame = feat_cluster_labels[frame_id]
            frame_covered = (covered_patches[frame_id]
                             if covered_patches is not None else None)
            patch_coords = _get_cluster_patch_coords(
                cluster_id, feat_labels_frame, frame_covered)
            if patch_coords is None:
                continue
            all_patch_coords.append(patch_coords)
            all_frame_ids.append(
                np.full(len(patch_coords), frame_id, dtype=np.int32))

        if not all_patch_coords:
            continue

        pooled_patches = np.concatenate(all_patch_coords, axis=0)  # (M, 2) y,x
        pooled_frames = np.concatenate(all_frame_ids, axis=0)       # (M,)

        # Run k-means on the full duplicated pool. Duplicate patch coords
        # act as implicit frequency weights — patches that appear in many
        # allocated frames pull centroids proportionally, giving better
        # temporal consistency. Only clamp k to prevent sklearn from
        # creating more clusters than distinct points (empty-cluster crash).
        n_unique = len(np.unique(pooled_patches, axis=0))
        effective_k = min(total_k, n_unique)
        selected_patches = cluster_coordinates(pooled_patches, effective_k)  # (K, 2)

        # Map each selected centroid-nearest patch back to its source frame
        tree = cKDTree(pooled_patches)
        _, indices = tree.query(selected_patches)
        sel_frames = pooled_frames[indices]

        # Jitter patch coords -> pixel coords with uniform offset inside
        # the patch cell. Prevents all keypoints landing at patch centers.
        jitter = rng.uniform(0.0, 1.0, size=selected_patches.shape)  # (K, 2) y,x
        sel_y = np.clip(
            (selected_patches[:, 0] + jitter[:, 0]) * cell_h, 0, h - 1)
        sel_x = np.clip(
            (selected_patches[:, 1] + jitter[:, 1]) * cell_w, 0, w - 1)
        sel_xy = np.stack([sel_x, sel_y], axis=1)  # (K, 2) x,y

        for i in range(len(sel_xy)):
            fid = int(sel_frames[i])
            frame_points[fid].append(sel_xy[i])
            frame_labels[fid].append(cluster_id)

    # --- Assemble per-frame outputs ---
    points_list = []
    point_labels_list = []
    component_labels_list = []
    cluster_frame_id_list = []

    for frame_id in sorted(frame_points.keys()):
        pts = np.array(frame_points[frame_id])        # (N, 2) x,y
        labs = np.array(frame_labels[frame_id])        # (N,)
        if len(pts) == 0:
            continue
        points_list.append(pts)
        point_labels_list.append(labs)
        if args.use_connected_components:
            component_labels_list.append(np.array(frame_comp_labels[frame_id]))
        cluster_frame_id_list.append(frame_id)

        if args.debug_mode:
            make_frame_cluster_vis(video_frames, frame_id, feat_cluster_labels,
                                   debug_vis_root)

    return points_list, point_labels_list, component_labels_list, cluster_frame_id_list
