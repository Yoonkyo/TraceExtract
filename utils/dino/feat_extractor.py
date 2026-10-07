import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from einops import rearrange
from .clustering import TorchKMeansVectorizedCluster


class feature_extract(nn.Module):
    def __init__(self):
        super().__init__()
        # Currently dinov2 base is hard coded
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available()
                                   else "cpu")
        self.dinov2.to(self.device)
        self.dinov2.eval()


    @torch.no_grad()
    def forward(self, frames, model_type='dino'):
        """
        Args:
            frames: numpy array of shape (bs, num_frames, height, width, channel).
            model_type: one of 'dino', 'clip', or 'siglip'
        """
        batch_size, num_frames, _, _, _ = frames.shape

        if model_type == 'dino':
            # DINO preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

            # Process all frames
            processed_frames = []
            for i in range(batch_size):
                for j in range(num_frames):
                    frame_pil = Image.fromarray(frames[i, j])
                    frame_tensor = transform(frame_pil)
                    processed_frames.append(frame_tensor)

            # Stack and reshape
            x = torch.stack(processed_frames)

            # Process in batches to avoid OOM with large frame counts
            dino_batch_size = 32
            if x.shape[0] <= dino_batch_size:
                x = x.to(self.device)
                feat = self.dinov2.forward_features(x)['x_norm_patchtokens']
            else:
                feat_parts = []
                for start in range(0, x.shape[0], dino_batch_size):
                    x_batch = x[start:start + dino_batch_size].to(self.device)
                    feat_batch = self.dinov2.forward_features(x_batch)['x_norm_patchtokens']
                    feat_parts.append(feat_batch)
                feat = torch.cat(feat_parts, dim=0)

        else:
            raise ValueError(f"Invalid model type: {model_type}")

        # Reshape features
        feat = rearrange(feat, '(b t) p d -> b t p d', b=batch_size, t=num_frames)
        patch_size = int(feat.shape[2] ** 0.5)
        feat = rearrange(feat, 'b t (p q) d -> b t p q d', p=patch_size)

        return feat

    def cluster_features(self, feat, method='dbscan', n_clusters=8,
                         prev_centers=None, global_clustering=False, use_torch=False):
        """
        Args:
            feat: Features to cluster
            method: Clustering method ('dbscan' or 'kmeans')
            n_clusters: Number of clusters for kmeans
            prev_centers: Previous cluster centers for temporal consistency
            global_clustering: If True, cluster all frames together
        """
        if global_clustering:
            # For global clustering, reshape to (n_frames * n_patches, n_features)
            n_frames, n_patches_h, n_patches_w, feat_dim = feat.shape

            if use_torch:
                feat_2d = feat.reshape(-1, feat_dim)
            else:
                feat_2d = feat.reshape(-1, feat_dim).cpu().numpy()

        else:
            # Original per-frame clustering
            n_patches = feat.shape[0] * feat.shape[1]

            if use_torch:
                feat_2d = feat.reshape(n_patches, -1)
            else:
                feat_2d = feat.reshape(n_patches, -1).cpu().numpy()

        if method == 'dbscan':
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(feat_2d)
            labels = clustering.labels_
            centers = None
        elif method == 'kmeans':
            if use_torch:
                device = feat_2d.device
                clustering = TorchKMeansVectorizedCluster(n_clusters=n_clusters)
                if prev_centers is not None:
                    prev_centers = torch.tensor(prev_centers).to(device)
                labels, centers = clustering(feat_2d, prev_centers=prev_centers)
                labels = labels.cpu().numpy()
                centers = centers.cpu().numpy()

            else:
                from sklearn.cluster import KMeans
                if prev_centers is not None:
                    clustering = KMeans(n_clusters=n_clusters, random_state=42,
                                            init=prev_centers, n_init=1)
                else:
                    clustering = KMeans(n_clusters=n_clusters, random_state=42)

                clustering.fit(feat_2d)
                labels = clustering.labels_
                centers = clustering.cluster_centers_
        else:
            raise ValueError("Method must be either 'dbscan' or 'kmeans'")

        if global_clustering:
            # Reshape labels back to (n_frames, patch_h, patch_w)
            labels = labels.reshape(n_frames, n_patches_h, n_patches_w)
        else:
            # Reshape labels back to patch grid
            labels = labels.reshape(feat.shape[0], feat.shape[1])

        return labels, centers
