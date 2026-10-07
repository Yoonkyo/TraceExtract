"""
DINO-based semantic keypoint extraction for TraceExtract.

Uses DINOv2 features and clustering to identify semantically meaningful
keypoints at peak-visibility frames, replacing uniform grid sampling.
"""
import numpy as np
import torch
from loguru import logger

from .dino.feat_extractor import feature_extract as DinoFeatureExtractor
from .dino.get_semantic_points import get_points_from_clustering


class DinoArgs:
    """Namespace adapter for Trace_gen functions that expect an ``args`` object."""

    def __init__(
        self,
        clustering_method='bipartite',
        merge_ratio=25,
        num_iters=11,
        n_clusters=32,
        num_points_per_entity=64,
        use_connected_components=False,
        debug_mode=False,
    ):
        self.clustering_method = clustering_method
        self.merge_ratio = merge_ratio
        self.num_iters = num_iters
        self.n_clusters = n_clusters
        self.num_points_per_entity = num_points_per_entity
        self.use_connected_components = use_connected_components
        self.debug_mode = debug_mode


def create_dino_extractor():
    """Create and return a DINOv2 ViT-B/14 feature extractor."""
    return DinoFeatureExtractor()


def extract_dino_keypoints(
    video_tensor,
    dino_extractor,
    clustering_method='bipartite',
    merge_ratio=25,
    clustering_num_iters=11,
    n_clusters=32,
    num_points_per_entity=64,
    use_connected_components=False,
    debug=False,
    debug_vis_root=None,
    covered_patches=None,
    dino_stride=2,
):
    """Extract semantically meaningful keypoints via DINO features + clustering.

    Args:
        video_tensor: ``(T, C, H, W)`` float tensor in ``[0, 1]``.
        dino_extractor: DINO feature extractor model instance.
        clustering_method: ``'bipartite'`` or ``'kmeans'``.
        merge_ratio: merge ratio for bipartite clustering.
        clustering_num_iters: number of iterations for bipartite clustering.
        n_clusters: number of clusters for k-means.
        num_points_per_entity: keypoints to sample per cluster.
        use_connected_components: use connected components analysis.
        debug: save debug cluster visualisations.
        debug_vis_root: directory for debug cluster visualisations.
        covered_patches: optional ``(T, 16, 16)`` bool array. ``True`` means
            the patch is already tracked and should be excluded from counting
            and point sampling.
        dino_stride: frame stride for DINO feature extraction (default 2).

    Returns:
        frame_data:
            ``dict[int, dict]`` mapping **original** frame index to
            ``{'points': (N, 2) xy, 'cluster_ids': (N,)}``.
        all_cluster_ids:
            sorted list of every unique cluster ID found.
    """
    T, C, H, W = video_tensor.shape

    # Subsample frames by dino_stride
    dino_frame_indices = list(range(0, T, dino_stride))

    logger.info(f"DINO: using {len(dino_frame_indices)}/{T} frames (stride={dino_stride})")

    dino_frames = video_tensor[dino_frame_indices]          # (T', C, H, W)
    dino_frames = dino_frames.permute(0, 2, 3, 1)          # (T', H, W, C)
    dino_frames = (dino_frames * 255).clamp(0, 255).byte().cpu().numpy()
    dino_frames = dino_frames[np.newaxis]                   # (1, T', H, W, C)

    # Subsample covered_patches to match dino_frame_indices
    dino_covered_patches = None
    if covered_patches is not None:
        dino_covered_patches = covered_patches[dino_frame_indices]

    dino_args = DinoArgs(
        clustering_method=clustering_method,
        merge_ratio=merge_ratio,
        num_iters=clustering_num_iters,
        n_clusters=n_clusters,
        num_points_per_entity=num_points_per_entity,
        use_connected_components=use_connected_components,
        debug_mode=debug,
    )

    if debug_vis_root is None:
        debug_vis_root = '/tmp/dino_debug'

    # --- DINO forward + clustering + proportional keypoint sampling ---
    (
        points_list,
        point_labels_list,
        component_labels_list,
        cluster_frame_id_list,
    ) = get_points_from_clustering(dino_args, dino_frames, dino_extractor, debug_vis_root,
                                   covered_patches=dino_covered_patches)

    # --- map subsampled indices back to original video indices ---
    frame_data = {}
    all_cluster_ids = set()

    for i, dino_local_idx in enumerate(cluster_frame_id_list):
        original_frame_idx = dino_frame_indices[dino_local_idx]
        points = np.asarray(points_list[i])             # (N_i, 2)  x, y
        cluster_ids = np.asarray(point_labels_list[i])   # (N_i,)

        all_cluster_ids.update(int(c) for c in cluster_ids)

        if original_frame_idx in frame_data:
            # Multiple subsampled frames can map here; merge points
            frame_data[original_frame_idx] = {
                'points': np.concatenate([frame_data[original_frame_idx]['points'], points], axis=0),
                'cluster_ids': np.concatenate([frame_data[original_frame_idx]['cluster_ids'], cluster_ids], axis=0),
            }
        else:
            frame_data[original_frame_idx] = {
                'points': points,
                'cluster_ids': cluster_ids,
            }

    logger.info(
        f"DINO: {len(frame_data)} query frames, "
        f"{sum(d['points'].shape[0] for d in frame_data.values())} total keypoints, "
        f"{len(all_cluster_ids)} clusters"
    )
    return frame_data, sorted(all_cluster_ids)
