import torch
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
import numpy as np

class TorchKMeansVectorizedCluster(nn.Module):
    def __init__(self, n_clusters=8, max_iters=300, tol=1e-4,
                                                n_init=1, spatial_weight=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init
        self.spatial_weight = spatial_weight
        self.centroids = None

    def _add_spatial_coordinates(self, X, H, W):
        """
        Add normalized spatial coordinates to feature vectors
        X: [N, D] where N = H*W
        Returns: [N, D+2] with x,y coordinates appended
        """
        assert X.shape[0] == H * W, f"Input shape does not match H*W={H*W}"

        device = X.device
        # Create normalized coordinate grid
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        spatial_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

        # Scale spatial coordinates by weight and append to features
        spatial_coords = spatial_coords * self.spatial_weight
        return torch.cat([X, spatial_coords], dim=1)

    @torch.no_grad()
    def _single_kmeans(self, X, prev_centers=None):
        """
        Single run of k-means
        X: [N, D]
        H, W: height and width of the feature grid (default 16x16)
        prev_centers: [n_clusters, D] (optional)
        """
        device = X.device
        # Add spatial coordinates
        # X_with_spatial = self._add_spatial_coordinates(X, H, W)
        X_with_spatial = X

        if prev_centers is not None:
            centroids = prev_centers.to(device)
        else:
            indices = torch.randperm(X_with_spatial.size(0),
                                      device=device)[: self.n_clusters]
            centroids = X_with_spatial[indices].clone()

        for _ in range(self.max_iters):
            old_centroids = centroids.clone()

            distances = torch.cdist(X_with_spatial, centroids, p=2)
            labels = torch.argmin(distances, dim=1)

            one_hot = F.one_hot(labels, num_classes=self.n_clusters).float()
            counts = one_hot.sum(dim=0).clamp_min(1.0)
            new_centroids = (one_hot.T @ X_with_spatial) / counts.unsqueeze(1)

            centroids = new_centroids

            move_distance = (old_centroids - centroids).abs().max().item()
            if move_distance < self.tol:
                break

        distances = torch.cdist(X_with_spatial, centroids, p=2)
        labels = torch.argmin(distances, dim=1)
        inertia = torch.sum(torch.min(distances, dim=1)[0])

        return labels, centroids, inertia

    @torch.no_grad()
    def forward(self, X, prev_centers=None):
        """
        X: [N, D]
        H, W: height and width of the feature grid (default 16x16)
        prev_centers: [n_clusters, D+2] (optional)
        """

        if prev_centers is not None:
            labels, centroids, _ = self._single_kmeans(X, prev_centers)
            self.centroids = centroids
            return labels, centroids

        best_inertia = float("inf")
        best_labels = None
        best_centroids = None

        for _ in range(self.n_init):
            labels, centroids, inertia = self._single_kmeans(X)

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids

        self.centroids = best_centroids
        return best_labels, best_centroids


def merge_temporal_centroids(centroids: torch.Tensor,
                             similarity_threshold: float = 0.2) -> torch.Tensor:
    """
    Optimized version of temporal centroid merging.

    Args:
        centroids: Tensor of shape [batch, time, num_centroids, embed_dim]
        similarity_threshold: Threshold for merging centroids (default: 0.5)

    Returns:
        merged_centroids: List of unique centroids for each batch
    """
    device = centroids.device
    batch_size, num_frames, num_centroids, embed_dim = centroids.shape
    batch_merged_centroids = []

    # Pre-normalize all centroids at once for the entire batch
    normalized_centroids = F.normalize(centroids.reshape(-1, embed_dim), p=2, dim=-1)
    normalized_centroids = normalized_centroids.reshape(
                            batch_size, num_frames, num_centroids, embed_dim)
    max_centroids = 0

    for batch_idx in range(batch_size):
        # Process each batch
        batch_centroids = centroids[batch_idx]
        batch_normalized = normalized_centroids[batch_idx]

        # Initialize with first frame's centroids
        unique_centroids = [batch_centroids[0]]
        unique_normalized = batch_normalized[0].unsqueeze(0)  # [1, num_centroids, embed_dim]

        for t in range(1, num_frames):
            current = batch_centroids[t]
            current_norm = batch_normalized[t]

            # Compute similarities with all existing centroids at once
            similarities = torch.mm(
                current_norm,
                unique_normalized.reshape(-1, embed_dim).t()
            ).reshape(num_centroids, -1)

            # Find max similarity for each current centroid
            max_sim, _ = similarities.max(dim=1)
            new_mask = max_sim < similarity_threshold

            if new_mask.any():
                # Add new unique centroids
                new_centroids = current[new_mask]
                unique_centroids.append(new_centroids)
                unique_normalized = torch.cat([
                    unique_normalized,
                    current_norm[new_mask].unsqueeze(0)
                ], dim=1)

        # Final concatenation for this batch
        unique_centroids = torch.cat(unique_centroids, dim=0)
        batch_merged_centroids.append(unique_centroids)
        max_centroids = max(max_centroids, len(unique_centroids))

    centroid_mask = torch.ones(batch_size, num_frames, max_centroids)

    for i in range(len(batch_merged_centroids)):
        current_num_centroids = len(batch_merged_centroids[i])
        if current_num_centroids < max_centroids:
            batch_merged_centroids[i] = torch.cat([
                batch_merged_centroids[i],
                torch.zeros(max_centroids - current_num_centroids,
                            batch_merged_centroids[i].shape[-1]).to(device)])
            centroid_mask[i, :, current_num_centroids:] = 0
        batch_merged_centroids[i] = batch_merged_centroids[i][None,...]
    batch_merged_centroids = torch.cat(batch_merged_centroids, dim=0)
    return batch_merged_centroids, centroid_mask.to(device)

def bipartite_matching_for_assignemnt(set_a: torch.Tensor, set_b: torch.Tensor,
                                      cluster_mask: torch.Tensor = None,
                                      temperature: float = 1.0) -> torch.Tensor:
    """
    Efficiently matches elements from set_a to set_b using cosine similarity.

    Args:
        set_a: Tensor of shape [batch, tokens_a, channels]
        set_b: Tensor of shape [batch, tokens_b, channels]
        temperature: Optional scaling factor for similarity scores (default: 1.0)
                    Higher values make matching more decisive/sharp

    Returns:
        matched_indices: Tensor of shape [batch, tokens_a] containing indices
                        of elements in set_b that each element in set_a matches to
    """
    # Normalize vectors in a single fused operation
    set_a_norm = torch.nn.functional.normalize(set_a, p=2, dim=-1)
    set_b_norm = torch.nn.functional.normalize(set_b, p=2, dim=-1)
    batch_size, set_a_tokens = set_a_norm.shape[:2]
    _, set_b_tokens = set_b_norm.shape[:2]

    # Compute similarity scores with optional temperature scaling
    # Using torch.baddbmm can be more efficient for batch matrix multiplication
    empty_tensor = torch.zeros((batch_size, set_a_tokens,
                                set_b_tokens)).to(set_a_norm.device)
    scores = torch.baddbmm(empty_tensor, set_a_norm,
                           set_b_norm.transpose(-1, -2),
                           beta=0.0, alpha=1.0/temperature)
    # Use torch.max with
    #  dim argument for efficient index selection
    if cluster_mask is not None:
        cluster_mask = cluster_mask.unsqueeze(1)
        scores = scores * cluster_mask
    max_sim, matched_indices = torch.max(scores, dim=-1)

    return matched_indices



def bipartite_soft_matching(k: torch.Tensor, r: int) -> torch.Tensor:
    """
    Input is k from attention, size [batch, tokens, channels].
    r is the reduction ratio
    """
    batch_size = k.shape[0]

    k = k / k.norm(dim=-1, keepdim=True)

    # Create index mapping for even and odd tokens
    num_tokens = k.shape[1]
    even_indices = list(range(0, num_tokens, 2))
    odd_indices = list(range(1, num_tokens, 2))

    # Split tokens while keeping track of original indices
    a, b = k[..., even_indices, :], k[..., odd_indices, :]

    scores = torch.matmul(a, b.transpose(-1, -2))

    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

    unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
    src_idx = edge_idx[..., :r, :]  # Merged Tokens
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor) -> torch.Tensor:
        """Input is of shape [batch, tokens, channels]."""
        # Split using the same even/odd indices
        src, dst = x[..., even_indices, :], x[..., odd_indices, :]
        n, t1, c = src.shape

        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))

        # Create a mask for positions that will be merged
        merge_mask = torch.zeros_like(dst)
        merge_mask.scatter_(-2, dst_idx.expand(n, r, c), 1.0)

        # Count how many tokens are merged into each position
        merge_counts = torch.zeros_like(dst[..., 0])
        merge_counts.scatter_add_(-1, dst_idx[..., 0],
                        torch.ones_like(src_idx[..., 0], dtype=torch.float32))
        merge_counts = merge_counts.unsqueeze(-1) + 1.0  # Add 1 for the destination token itself
        # Add the features
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src)
        # Divide by the actual number of merged tokens
        dst = dst * torch.where(merge_mask > 0, 1.0/merge_counts, 1.0)

        return torch.cat([unm, dst], dim=-2)

    return merge


def get_bipartite_soft_matching(patch_tokens, merge_ratio, iter=12):
    all_tokens = patch_tokens.shape[1]
    for _ in range(iter):
        num_tokens = patch_tokens.shape[1]
        num_merge_tokens = int(num_tokens * merge_ratio / 100)
        merge_fn = bipartite_soft_matching(patch_tokens.clone(), num_merge_tokens)
        patch_tokens = merge_fn(patch_tokens)
    return patch_tokens

def get_temporal_bipartite_clusters(patch_tokens, merge_ratio=25, num_iters=11):
    """
    Modified to use bipartite matching for clustering
    Args:
        patch_tokens: tensor of shape [T, P, Q, D]
    Returns:
        cluster_ids: tensor of shape [B, T, P]
    """

    device = patch_tokens.device
    t, p, q, d  = patch_tokens.shape
    patch_tokens = rearrange(patch_tokens, 't p q d -> 1 t (p q) d')
    batch_frame_tokens = rearrange(patch_tokens, 'b t p d -> (b t) p d')
    cluster_features = get_bipartite_soft_matching(batch_frame_tokens,
                                                   merge_ratio, iter=num_iters)
    cluster_features = rearrange(cluster_features, '(b t) p d -> b t p d', t=t)
    cluster_features, centroid_mask = merge_temporal_centroids(cluster_features)
    num_frames = patch_tokens.shape[1]
    cluster_features = cluster_features.unsqueeze(1).repeat(1, num_frames, 1, 1)
    cluster_features = rearrange(cluster_features, 'b t c d -> (b t) c d')
    centroid_mask = rearrange(centroid_mask, 'b t c -> (b t) c')

    patch_tokens = rearrange(patch_tokens, 'b t p d -> (b t) p d')
    cluster_ids = bipartite_matching_for_assignemnt(
                                patch_tokens, cluster_features, centroid_mask)
    cluster_ids = rearrange(cluster_ids, 't (p q) -> t p q', t=num_frames, p=p, q=q)
    return cluster_ids.long().cpu().numpy()


def cluster_coordinates(coords: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster coordinates based on their spatial positions using K-means and find points
    closest to cluster centers.

    Args:
        coords: Array of shape (N, 2) containing (x, y) coordinates
        n_clusters: Number of clusters to create

    Returns:
        tuple of:
            labels: Array of shape (N,) containing cluster assignments
            center_points: Array of shape (n_clusters, 2) containing coordinates of points
                         closest to each cluster center
    """
    from sklearn.cluster import KMeans

    # Ensure coords is the right shape
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be of shape (N, 2), got {coords.shape}")

    # Handle case where N < n_clusters
    if len(coords) < n_clusters:
        # Assign each point to its own cluster
        labels = np.arange(len(coords))
        # Use the actual points as centers for available clusters
        center_points = coords
        return center_points

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(coords)
    centroids = kmeans.cluster_centers_

    # Find points closest to each centroid
    center_points = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        # Get points in this cluster
        cluster_points = coords[labels == i]
        # Calculate distances to centroid
        distances = np.sqrt(np.sum((cluster_points - centroids[i]) ** 2, axis=1))
        # Find the closest point
        closest_idx = np.argmin(distances)
        center_points[i] = cluster_points[closest_idx]

    return center_points


def cluster_coordinates_per_component(components, num_points_total, cluster_coordinates_fn):
    """Cluster coordinates within each connected component.

    Args:
        components: List of arrays containing (y,x) coordinates for each component
        num_points_total: Total number of points to sample across all components
        cluster_coordinates_fn: Function to cluster coordinates within a component

    Returns:
        clustered_points: Array of shape (N, 2) containing final sampled points
    """
    # Calculate points per component proportional to component size
    total_pixels = sum(len(comp) for comp in components)
    points_per_component = []

    for comp in components:
        points_per_component.append(num_points_total)

    all_clustered_points = []
    point_component_labels =[]
    for component_idx, (component, n_points) in enumerate(zip(components, points_per_component)):
        if len(component) <= n_points:
            clustered_points = component
        else:
            clustered_points = cluster_coordinates_fn(component, n_points)
        all_clustered_points.append(clustered_points)
        point_component_labels.append(np.ones(len(clustered_points)) * component_idx)
    return np.vstack(all_clustered_points), np.concatenate(point_component_labels).astype(int)
