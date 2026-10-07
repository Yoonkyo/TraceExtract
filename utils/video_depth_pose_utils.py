import os
import math
import torch
import numpy as np
from loguru import logger

from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from utils.se3_utils import align_chunk_to_global


def align_depth_scale(pred_depth, known_depth):
    """
    Align the scale of predicted depth to known depth using median scaling.
    Input:
        pred_depth: (H, W), torch tensor
        known_depth: (H, W), torch tensor
    Return:
        aligned_depth: (H, W), torch tensor
    """
    valid_mask = (known_depth > 0) & (pred_depth > 0)
    scale = np.median(known_depth[valid_mask]) / np.median(pred_depth[valid_mask])
    return scale

def align_video_depth_scale(pred_depth, known_depth):
    """
    Align the scale of predicted depth to known depth using median scaling.
    Input:
        pred_depth: (T, H, W), torch tensor
        known_depth: (T, H, W), torch tensor
    Return:
        aligned_depth: (T, H, W), torch tensor
    """
    scales = []
    for t in range(pred_depth.shape[0]):
        scales.append(
            align_depth_scale(pred_depth[t], known_depth[t])
        )
    scale = np.array(scales).mean()
    aligned_depth = pred_depth * scale
    return aligned_depth, scale


class BaseVideoDepthPoseWrapper:
    def __init__(self, args):
        self.args = args
        self.device = args.device

    def __call__(self, video_tensor, known_depth=None, stationary_camera=False, replace_with_known_depth=True):
        """
        Input:
            video_tensor: (T, 3, H, W), torch tensor, range [0, 1]
            known_depth: (T, H, W), torch tensor, range [0, inf), if provided, will be used as depth input
            stationary_camera: bool, if True, indicates the camera is stationary
            replace_with_known_depth: bool, only used when known_depth is provided. If True, return known_depth as the depth. If False, rescale the estimated depth to the scale of known depth
        Return:
            video_ten: (T, 3, H, W), torch tensor, range [0, 1], processed (resized) video tensor, will be used in 3D tracking as well
            depth_npy: (T, H, W), numpy array
            depth_conf: (T, H, W), numpy array
            extrs_npy: (T, 4, 4), numpy array, in camera-to-world format
            intrs_npy: (T, 3, 3), numpy array
        """
        raise NotImplementedError


class VGGT4Wrapper(BaseVideoDepthPoseWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.load_model()

    def load_model(self, checkpoint_path="Yuxihenry/SpatialTrackerV2_Front"):
        model = VGGT4Track.from_pretrained(checkpoint_path)
        logger.debug(f"load vggt4 from {checkpoint_path}")
        model = model.eval()
        model = model.to(self.device)
        return model

    def to_device(self, device):
        """Move model to a device (for memory management)."""
        self.model = self.model.to(device)
        if device == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def __call__(self, video_tensor, known_depth=None, stationary_camera=False, replace_with_known_depth=True):
        video_tensor_processed = preprocess_image(video_tensor)[None]  # (1, T, 3, H, W)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                predictions = self.model(video_tensor_processed.cuda())
                extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
                depth_map, depth_conf = (
                    predictions["points_map"][..., 2],
                    predictions["unc_metric"],
                )

        depth_npy = depth_map.squeeze().cpu().numpy()
        extrs_npy = extrinsic.squeeze().cpu().numpy()
        intrs_npy = intrinsic.squeeze().cpu().numpy()
        video_ten = video_tensor_processed.squeeze()

        if known_depth is not None:
            known_depth = torch.nn.functional.interpolate(
                known_depth[:, None, :, :], size=depth_npy.shape[1:], mode="bilinear", align_corners=False
            )[:, 0, :, :]
            known_depth = known_depth.cpu().numpy()
            depth_npy, scale = align_video_depth_scale(
                depth_npy, known_depth
            )
            if replace_with_known_depth:
                depth_npy = known_depth
                depth_conf = (known_depth > 0).astype(np.float32)

            extrs_npy[:, :3, 3] *= scale

        if stationary_camera:
            extrs_npy = np.repeat(extrs_npy[0:1], extrs_npy.shape[0], axis=0)

        return video_ten, depth_npy, depth_conf, extrs_npy, intrs_npy

    # ------------------------------------------------------------------
    #  Hybrid chunked inference: global sparse + local dense
    # ------------------------------------------------------------------

    def _run_vggt_on_frames(self, video_tensor):
        """Run VGGT on a (T, 3, H, W) tensor and return raw numpy outputs.

        Returns:
            video_processed: (T, 3, H', W') preprocessed tensor (on CPU)
            depth_npy: (T, H', W')
            depth_conf: (T, H', W')  — raw tensor on CPU
            extrs_npy: (T, 4, 4) camera-to-world
            intrs_npy: (T, 3, 3)
        """
        video_tensor_processed = preprocess_image(video_tensor)[None]  # (1, T, 3, H', W')

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                predictions = self.model(video_tensor_processed.cuda())
                extrinsic = predictions["poses_pred"]
                intrinsic = predictions["intrs"]
                depth_map = predictions["points_map"][..., 2]
                depth_conf = predictions["unc_metric"]

        depth_npy = depth_map.squeeze().cpu().numpy()
        extrs_npy = extrinsic.squeeze().cpu().numpy()
        intrs_npy = intrinsic.squeeze().cpu().numpy()
        video_processed = video_tensor_processed.squeeze().cpu()
        depth_conf_npy = depth_conf.squeeze().cpu().numpy()

        return video_processed, depth_npy, depth_conf_npy, extrs_npy, intrs_npy

    def chunked_inference(
        self,
        video_tensor,
        chunk_size=150,
        sparse_max=150,
        known_depth=None,
        stationary_camera=False,
        replace_with_known_depth=True,
    ):
        """Hybrid global-sparse + local-dense VGGT inference for long videos.

        1. Uniformly subsample to ≤ sparse_max frames → global sparse VGGT pass
           → globally consistent anchor extrinsics + averaged intrinsics.
        2. Split full video into non-overlapping chunks of chunk_size frames.
        3. Per chunk: run VGGT → align to global frame via anchor pairs.
        4. Assemble globally consistent depth, extrinsics, intrinsics.

        Short-video fast path: if T ≤ chunk_size, just calls __call__().

        Args:
            video_tensor: (T, 3, H, W) float tensor in [0, 1].
            chunk_size: max frames per dense VGGT chunk.
            sparse_max: max frames for the global sparse pass.
            known_depth: optional (T, H, W) tensor.
            stationary_camera: if True, copy first-frame extrinsics to all.
            replace_with_known_depth: whether to replace depth with known.

        Returns:
            Same 5-tuple as __call__():
            video_ten, depth_npy, depth_conf, extrs_npy, intrs_npy
        """
        T_total = video_tensor.shape[0]

        # ── Fast path: short video ──
        if T_total <= chunk_size:
            logger.info(
                f"VGGT: short video ({T_total} frames ≤ chunk_size={chunk_size}), "
                f"using single-pass inference"
            )
            return self(
                video_tensor,
                known_depth=known_depth,
                stationary_camera=stationary_camera,
                replace_with_known_depth=replace_with_known_depth,
            )

        logger.info(
            f"VGGT chunked inference: {T_total} frames, "
            f"chunk_size={chunk_size}, sparse_max={sparse_max}"
        )

        # ==================================================================
        #  Step 1: Global sparse pass
        # ==================================================================
        n_sparse = min(T_total, sparse_max)
        sparse_indices = np.linspace(0, T_total - 1, n_sparse, dtype=int)
        sparse_indices = np.unique(sparse_indices)  # deduplicate
        sparse_video = video_tensor[sparse_indices]

        logger.info(f"VGGT sparse pass: {len(sparse_indices)} frames")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        (
            _sparse_video_proc,
            _sparse_depth,
            _sparse_depth_conf,
            sparse_extrs,
            sparse_intrs,
        ) = self._run_vggt_on_frames(sparse_video)

        # ==================================================================
        #  Step 2: Global intrinsics — average from sparse pass
        # ==================================================================
        global_K = sparse_intrs.mean(axis=0)  # (3, 3)
        logger.info(f"VGGT: averaged global intrinsics from {len(sparse_intrs)} sparse frames")

        # ==================================================================
        #  Step 3: Dense chunks — non-overlapping
        # ==================================================================
        n_chunks = math.ceil(T_total / chunk_size)
        chunk_ranges = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, T_total)
            if start < end:
                chunk_ranges.append((start, end))

        # A stray 1-frame tail chunk makes _run_vggt_on_frames silently squeeze
        # away the time dimension, which later breaks anchor-based SE3
        # alignment with a shape mismatch (sources (1,4) vs targets (1,4,4)).
        # Merge it into the previous chunk instead of processing it alone.
        if len(chunk_ranges) >= 2 and (chunk_ranges[-1][1] - chunk_ranges[-1][0]) == 1:
            prev_start, _ = chunk_ranges[-2]
            _, last_end = chunk_ranges[-1]
            chunk_ranges[-2] = (prev_start, last_end)
            chunk_ranges.pop()

        logger.info(f"VGGT dense passes: {len(chunk_ranges)} chunk(s)")

        # Pre-allocate output arrays (we'll fill after first chunk to get H', W')
        video_processed_list = []
        depth_list = []
        depth_conf_list = []
        extrs_aligned_list = []

        for chunk_idx, (start, end) in enumerate(chunk_ranges):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            chunk_video = video_tensor[start:end]
            logger.info(
                f"  VGGT chunk {chunk_idx}: frames [{start}, {end}) "
                f"({end - start} frames)"
            )

            (
                chunk_video_proc,
                chunk_depth,
                chunk_depth_conf,
                chunk_extrs,
                chunk_intrs,
            ) = self._run_vggt_on_frames(chunk_video)

            # ── Find anchor pairs: (chunk_local_idx, sparse_idx) ──
            anchor_pairs = []
            for sparse_idx, global_frame in enumerate(sparse_indices):
                if start <= global_frame < end:
                    chunk_local_idx = global_frame - start
                    anchor_pairs.append((chunk_local_idx, sparse_idx))

            logger.debug(
                f"  Chunk {chunk_idx}: {len(anchor_pairs)} anchor pair(s) "
                f"for alignment"
            )

            # ── Align chunk extrinsics to global frame ──
            if len(anchor_pairs) > 0:
                chunk_extrs_aligned = align_chunk_to_global(
                    chunk_extrs, sparse_extrs, anchor_pairs,
                    outlier_sigma=2.0,
                )
            else:
                logger.warning(
                    f"  Chunk {chunk_idx}: no anchor pairs! "
                    f"Using unaligned extrinsics."
                )
                chunk_extrs_aligned = chunk_extrs

            video_processed_list.append(chunk_video_proc)
            depth_list.append(chunk_depth)
            depth_conf_list.append(chunk_depth_conf)
            extrs_aligned_list.append(chunk_extrs_aligned)

        # ==================================================================
        #  Step 4: Assemble global arrays
        # ==================================================================
        video_ten = torch.cat(video_processed_list, dim=0)      # (T_total, 3, H', W')
        depth_npy = np.concatenate(depth_list, axis=0)           # (T_total, H', W')
        depth_conf = np.concatenate(depth_conf_list, axis=0)     # (T_total, H', W')
        extrs_npy = np.concatenate(extrs_aligned_list, axis=0)   # (T_total, 4, 4)
        intrs_npy = np.tile(global_K, (T_total, 1, 1))           # (T_total, 3, 3)

        # ==================================================================
        #  Step 5: Handle known depth & stationary camera (same as __call__)
        # ==================================================================
        if known_depth is not None:
            known_depth = torch.nn.functional.interpolate(
                known_depth[:, None, :, :],
                size=depth_npy.shape[1:],
                mode="bilinear",
                align_corners=False,
            )[:, 0, :, :]
            known_depth = known_depth.cpu().numpy()
            depth_npy, scale = align_video_depth_scale(depth_npy, known_depth)
            if replace_with_known_depth:
                depth_npy = known_depth
                depth_conf = (known_depth > 0).astype(np.float32)
            extrs_npy[:, :3, 3] *= scale

        if stationary_camera:
            extrs_npy = np.repeat(extrs_npy[0:1], T_total, axis=0)

        logger.info(
            f"VGGT chunked inference done: {T_total} frames assembled "
            f"({len(chunk_ranges)} dense chunks, {len(sparse_indices)} sparse anchors)"
        )

        return video_ten, depth_npy, depth_conf, extrs_npy, intrs_npy


video_depth_pose_dict = {
    "vggt4": VGGT4Wrapper,
}
