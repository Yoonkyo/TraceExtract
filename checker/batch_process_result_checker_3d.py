#!/usr/bin/env python3
"""
Batch Process Result Checker

This script checks the output of batch processing for point tracking results.
It validates the directory structure, checks data integrity, and creates
visualizations of keypoints and trajectories overlaid on images.

Expected structure (the current unified format; the legacy per-frame
images/*.png + samples/*.npz format is also accepted):
<output_root>/
  <video_name>/
    images.npy, depth.npy, cameras.npz
    samples/
      frame_indices.npy, offsets.npy, keypoints.npy, traj.npy, ...

See docs/output_structure.md for the full schema.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.output_format import (
    is_unified_format, has_unified_samples, load_image, load_depth,
    load_sample, list_sample_frame_indices, get_image_shape,
)


class BatchProcessChecker:
    def __init__(self, output_root, max_videos_to_check=3, max_samples_per_video=3, visualize_history=False, use_raw=False, visualize_horizon=8):
        """
        Initialize the checker.

        Args:
            output_root (str): Root directory containing batch processing results
            max_videos_to_check (int): Maximum number of videos to check
            max_samples_per_video (int): Maximum number of samples per video to visualize
            visualize_history (bool): If True, visualize traj_history instead of traj
            use_raw (bool): If True, visualize raw_traj/raw_traj_history (pre-retarget)
                instead of the arc-length retargeted traj.
        """
        self.output_root = Path(output_root)
        self.max_videos_to_check = max_videos_to_check
        self.max_samples_per_video = max_samples_per_video
        self.visualize_history = visualize_history
        self.use_raw = use_raw
        self.errors = []
        self.warnings = []
        self.visualize_horizon = visualize_horizon

    def log_error(self, message):
        """Log an error message."""
        self.errors.append(message)
        print(f"❌ ERROR: {message}")

    def log_warning(self, message):
        """Log a warning message."""
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")

    def log_info(self, message):
        """Log an info message."""
        print(f"ℹ️  {message}")

    def log_success(self, message):
        """Log a success message."""
        print(f"✅ {message}")

    def check_overall_structure(self):
        """Check the overall directory structure."""
        print("\n" + "="*60)
        print("CHECKING OVERALL STRUCTURE")
        print("="*60)

        if not self.output_root.exists():
            self.log_error(f"Output root directory does not exist: {self.output_root}")
            return False

        # Find all video directories
        video_dirs = [d for d in self.output_root.iterdir() if d.is_dir()]

        if not video_dirs:
            self.log_error(f"No video directories found in {self.output_root}")
            return False

        self.log_success(f"Found {len(video_dirs)} video directories")

        # Check structure for each video directory
        valid_videos = []
        for video_dir in video_dirs:
            if self.check_video_structure(video_dir):
                valid_videos.append(video_dir)

        if not valid_videos:
            self.log_error("No valid video directories found")
            return False

        self.log_success(f"{len(valid_videos)}/{len(video_dirs)} video directories have valid structure")
        return True

    def check_video_structure(self, video_dir):
        """Check structure for a single video directory."""
        video_name = video_dir.name
        vdir = str(video_dir)

        # ---- Unified .npy format -------------------------------------------
        if is_unified_format(vdir):
            images_npy = video_dir / "images.npy"
            samples_dir = video_dir / "samples"
            if not samples_dir.exists() or not (samples_dir / "frame_indices.npy").exists():
                self.log_warning(f"Missing samples/*.npy in {video_name}")
                return False
            T, H, W = get_image_shape(vdir, video_name)
            n_sample_frames = len(list_sample_frame_indices(vdir, video_name))
            self.log_info(f"Video {video_name} [npy]: {T} frames ({H}x{W}), {n_sample_frames} sample frames")
            return True

        # ---- Legacy per-frame format ----------------------------------------
        images_dir = video_dir / "images"
        samples_dir = video_dir / "samples"

        if not images_dir.exists():
            self.log_warning(f"Missing images directory in {video_name}")
            return False

        if not samples_dir.exists():
            self.log_warning(f"Missing samples directory in {video_name}")
            return False

        image_files = list(images_dir.glob("*.png"))
        sample_files = list(samples_dir.glob("*.npz"))

        if not image_files:
            self.log_warning(f"No PNG files found in {video_name}/images/")
            return False

        if not sample_files:
            self.log_warning(f"No NPZ files found in {video_name}/samples/")
            return False

        self.log_info(f"Video {video_name} [legacy]: {len(image_files)} images, {len(sample_files)} samples")
        return True

    def load_and_validate_sample_data(self, data, label):
        """Validate a sample data dict. *label* is used for log messages."""
        try:
            required_fields = ['keypoints', 'traj', 'valid_steps', 'frame_index']
            missing_fields = [f for f in required_fields if f not in data]

            if missing_fields:
                self.log_warning(f"Missing fields in {label}: {missing_fields}")
                return None

            keypoints = data['keypoints']
            traj = data['traj']

            if keypoints.ndim != 2 or keypoints.shape[1] != 2:
                self.log_warning(f"Invalid keypoints shape in {label}: {keypoints.shape}")
                return None

            if traj.ndim != 3 or traj.shape[0] != len(keypoints) or traj.shape[2] != 3:
                self.log_warning(f"Invalid trajectory shape in {label}: {traj.shape} (expected 3D with x,y,depth)")
                return None

            return data

        except Exception as e:
            self.log_error(f"Failed to validate sample {label}: {e}")
            return None

    def load_and_validate_sample(self, sample_path):
        """Load and validate a legacy per-frame sample file."""
        try:
            data_npz = np.load(sample_path)
            data = {k: data_npz[k] for k in data_npz.files}
            data_npz.close()
            return self.load_and_validate_sample_data(data, sample_path.name)
        except Exception as e:
            self.log_error(f"Failed to load sample {sample_path.name}: {e}")
            return None

    def check_keypoints_in_bounds(self, keypoints, image_shape, sample_name):
        """Check if keypoints are within image bounds."""
        h, w = image_shape[:2]

        # Check x coordinates (width)
        x_coords = keypoints[:, 0]
        x_out_of_bounds = np.logical_or(x_coords < 0, x_coords >= w)

        # Check y coordinates (height)
        y_coords = keypoints[:, 1]
        y_out_of_bounds = np.logical_or(y_coords < 0, y_coords >= h)

        out_of_bounds = np.logical_or(x_out_of_bounds, y_out_of_bounds)

        if np.any(out_of_bounds):
            n_out = np.sum(out_of_bounds)
            self.log_warning(f"{sample_name}: {n_out}/{len(keypoints)} keypoints out of bounds")
            return False
        else:
            self.log_info(f"{sample_name}: All keypoints within bounds")
            return True

    def create_visualization(self, image_or_path, data, output_path, video_name, frame_idx):
        """Create visualization with keypoints and trajectories overlaid.

        *image_or_path* can be an ndarray (H, W, 3) or a file path.
        """
        try:
            if isinstance(image_or_path, np.ndarray):
                image = image_or_path
            else:
                image = np.array(Image.open(image_or_path))
            h, w = image.shape[:2]

            keypoints = data['keypoints']
            if self.use_raw:
                # Pre-retarget per-frame trajectories
                if self.visualize_history and 'raw_traj_history' in data:
                    traj = data['raw_traj_history']
                    valid_steps_raw = data.get('raw_valid_steps_history',
                                               data.get('valid_steps_history', data['valid_steps']))
                else:
                    traj = data['raw_traj']
                    valid_steps_raw = data.get('raw_valid_steps', data['valid_steps'])
            else:
                if self.visualize_history and 'traj_history' in data:
                    traj = data['traj_history']  # [K, T, 3] - past trajectories
                    valid_steps_raw = data.get('valid_steps_history', data['valid_steps'])
                else:
                    traj = data['traj']  # [K, T, 3] - future trajectories
                    valid_steps_raw = data['valid_steps']

            # Per-track masking: mask now has shape (N, T). Older files may
            # still carry a 1D (T,) mask — broadcast it to the per-track form.
            valid_mask_arr = np.asarray(valid_steps_raw)
            if valid_mask_arr.ndim == 1:
                valid_mask_arr = np.broadcast_to(
                    valid_mask_arr.astype(bool), (traj.shape[0], traj.shape[1])
                ).copy()
            else:
                valid_mask_arr = valid_mask_arr.astype(bool)
            valid_steps = int(valid_mask_arr.any(axis=0).sum())
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Use a single color for all keypoints
            keypoint_color = plt.cm.tab20(0)

            # Plot 1: Keypoints overlay
            ax1.imshow(image)
            ax1.set_title(f'{video_name} Frame {frame_idx} - Keypoints')

            for kp in keypoints:
                ax1.scatter(kp[0], kp[1], c=[keypoint_color], s=50, alpha=0.8, edgecolors='white', linewidths=1)

            ax1.set_xlim(0, w)
            ax1.set_ylim(h, 0)  # Flip y-axis for image coordinates
            ax1.axis('off')

            # Plot 2: Trajectories overlay with depth-based coloring
            ax2.imshow(image)
            ax2.set_title(f'{video_name} Frame {frame_idx} - 3D Trajectories (Depth as Color)')
            for i, (kp, trajectory) in enumerate(zip(keypoints, traj)):
                # Plot current keypoint
                ax2.scatter(kp[0], kp[1], c=['red'], s=100, alpha=0.9,
                          edgecolors='white', linewidths=2, marker='o')

                # Per-track validity mask: row i of valid_mask_arr
                track_mask = valid_mask_arr[i]
                valid_traj = trajectory[track_mask]

                # Filter out invalid points (marked with -inf or nan) so they
                # never reach the projection / plotting code.
                if len(valid_traj) > 0:
                    finite_mask = ~(np.isinf(valid_traj).any(axis=1) | np.isnan(valid_traj).any(axis=1))
                    if np.any(finite_mask):
                        valid_traj_filtered = valid_traj[finite_mask][:self.visualize_horizon]

                        if len(valid_traj_filtered) > 1:
                            # Extract x, y coordinates and depth values
                            x_coords = valid_traj_filtered[:, 0]
                            y_coords = valid_traj_filtered[:, 1]
                            depth_values = valid_traj_filtered[:, 2]

                            # Normalize depth values for color mapping
                            if len(depth_values) > 0 and not np.all(np.isnan(depth_values)):
                                # Remove any remaining nan values in depth
                                valid_depth_mask = ~np.isnan(depth_values)
                                if np.any(valid_depth_mask):
                                    valid_depths = depth_values[valid_depth_mask]
                                    depth_min, depth_max = valid_depths.min(), valid_depths.max()

                                    if depth_max > depth_min:
                                        # Global normalization for consistent coloring
                                        depth_normalized = np.full_like(depth_values, 0.5)  # default middle value
                                        depth_normalized[valid_depth_mask] = (valid_depths - depth_min) / (depth_max - depth_min)
                                    else:
                                        depth_normalized = np.ones_like(depth_values) * 0.5

                                    # Create line segments with colors based on depth
                                    for j in range(len(x_coords) - 1):
                                        # Skip if either point has invalid depth
                                        if np.isnan(depth_normalized[j]) or np.isnan(depth_normalized[j+1]):
                                            segment_color = 'blue'
                                        else:
                                            # Use average depth of segment for color
                                            segment_depth = (depth_normalized[j] + depth_normalized[j+1]) / 2
                                            segment_color = plt.cm.viridis(segment_depth)

                                        ax2.plot([x_coords[j], x_coords[j+1]],
                                               [y_coords[j], y_coords[j+1]],
                                               color=segment_color, alpha=0.8, linewidth=2)

                                    # Mark each waypoint with an 'x' colored by its depth,
                                    # so the spacing from retargeting is visible.
                                    pt_colors = [plt.cm.viridis(d) if not np.isnan(d) else (1, 0, 0, 1)
                                                 for d in depth_normalized]
                                    ax2.scatter(x_coords, y_coords,
                                              c=pt_colors, s=50, alpha=0.9, marker='x', linewidths=2)
                                else:
                                    # All depth values are invalid
                                    ax2.plot(x_coords, y_coords, color='blue', alpha=0.7, linewidth=2)
                                    ax2.scatter(x_coords[:], y_coords[:],
                                              c=['blue'], s=50, alpha=0.7, marker='x', linewidths=2)
                            else:
                                # No valid depth values
                                ax2.plot(x_coords, y_coords, color='blue', alpha=0.7, linewidth=2)
                                ax2.scatter(x_coords[:], y_coords[:],
                                          c=['blue'], s=50, alpha=0.7, marker='x', linewidths=2)

            ax2.set_xlim(0, w)
            ax2.set_ylim(h, 0)  # Flip y-axis for image coordinates
            ax2.axis('off')

            # Add colorbar for depth visualization
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax2, shrink=0.8, aspect=20)
            cbar.set_label('Normalized Depth (0=Near, 1=Far)', rotation=270, labelpad=15)

            # Add info text
            info_text = f"Keypoints: {len(keypoints)}\nValid steps: {valid_steps}"
            fig.text(0.02, 0.98, info_text, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.log_success(f"Created visualization: {output_path}")
            return True

        except Exception as e:
            self.log_error(f"Failed to create visualization for {video_name} frame {frame_idx}: {e}")
            return False

    def check_and_visualize_data(self):
        """Check data integrity and create visualizations."""
        print("\n" + "="*60)
        print("CHECKING DATA AND CREATING VISUALIZATIONS")
        print("="*60)

        # Find video directories
        video_dirs = [d for d in self.output_root.iterdir() if d.is_dir()]
        video_dirs = sorted(video_dirs)[:self.max_videos_to_check]

        # Create output directory for visualizations
        vis_output_dir = self.output_root / "visualizations"
        vis_output_dir.mkdir(exist_ok=True)

        for video_dir in video_dirs:
            video_name = video_dir.name
            vdir = str(video_dir)
            self.log_info(f"Processing video: {video_name}")

            # Get frame indices with sample data (works for both formats)
            frame_indices = list_sample_frame_indices(vdir, video_name)
            if not frame_indices:
                continue
            frame_indices = frame_indices[:self.max_samples_per_video]

            for frame_idx in frame_indices:
                try:
                    data = load_sample(vdir, video_name, frame_idx)
                except (KeyError, FileNotFoundError) as e:
                    self.log_warning(str(e))
                    continue

                data = self.load_and_validate_sample_data(
                    data, f"{video_name}_{frame_idx}")
                if data is None:
                    continue

                try:
                    image = load_image(vdir, video_name, frame_idx)
                except FileNotFoundError:
                    self.log_warning(f"Missing image for {video_name} frame {frame_idx}")
                    continue

                try:
                    self.check_keypoints_in_bounds(
                        data['keypoints'], image.shape, f"{video_name}_{frame_idx}")

                    vis_filename = f"{video_name}_frame_{frame_idx}_visualization.png"
                    vis_path = vis_output_dir / vis_filename

                    self.create_visualization(
                        image, data, vis_path, video_name, frame_idx)

                except Exception as e:
                    self.log_error(f"Failed to process {video_name} frame {frame_idx}: {e}")
                    continue

    def generate_summary_report(self):
        """Generate a summary report."""
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)

        print(f"Output directory: {self.output_root}")

        # Count directories and files
        video_dirs = [d for d in self.output_root.iterdir() if d.is_dir() and d.name != "visualizations"]
        total_images = 0
        total_samples = 0

        for video_dir in video_dirs:
            vdir = str(video_dir)
            video_name = video_dir.name
            if is_unified_format(vdir):
                T, _, _ = get_image_shape(vdir, video_name)
                total_images += T
                total_samples += len(list_sample_frame_indices(vdir, video_name))
            else:
                images_dir = video_dir / "images"
                if images_dir.exists():
                    total_images += len(list(images_dir.glob("*.png")))
                samples_dir = video_dir / "samples"
                if samples_dir.exists():
                    total_samples += len(list(samples_dir.glob("*.npz")))

        print(f"Total videos: {len(video_dirs)}")
        print(f"Total images: {total_images}")
        print(f"Total samples: {total_samples}")
        print(f"Videos checked: {min(len(video_dirs), self.max_videos_to_check)}")

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\n🎉 All checks passed successfully!")

        # Check if visualizations were created
        vis_dir = self.output_root / "visualizations"
        if vis_dir.exists():
            vis_files = list(vis_dir.glob("*.png"))
            print(f"\n📊 Created {len(vis_files)} visualization files in: {vis_dir}")

    def run_checks(self):
        """Run all checks."""
        print("🔍 Starting batch process result checker...")

        # Step 1: Check overall structure
        if not self.check_overall_structure():
            self.log_error("Structure check failed. Stopping.")
            self.generate_summary_report()
            return False

        # Step 2: Check data and create visualizations
        self.check_and_visualize_data()

        # Step 3: Generate summary
        self.generate_summary_report()

        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Check batch processing results")
    parser.add_argument("output_root", help="Root directory containing batch processing results")
    parser.add_argument("--max-videos", type=int, default=3,
                       help="Maximum number of videos to check (default: 3)")
    parser.add_argument("--max-samples", type=int, default=3,
                       help="Maximum number of samples per video to visualize (default: 3)")
    parser.add_argument("--visualize_history", action="store_true",
                       help="Visualize traj_history (past trajectories) instead of traj (future)")
    parser.add_argument("--raw", action="store_true",
                       help="Visualize raw_traj/raw_traj_history (pre-retarget, frame-aligned) "
                            "instead of the arc-length retargeted traj.")
    parser.add_argument("--visualize_horizon", type=int, default=8,
                        help="Set how many timesteps we will visualize")

    args = parser.parse_args()

    checker = BatchProcessChecker(
        args.output_root,
        max_videos_to_check=args.max_videos,
        max_samples_per_video=args.max_samples,
        visualize_history=args.visualize_history,
        use_raw=args.raw,
        visualize_horizon=args.visualize_horizon
    )

    success = checker.run_checks()

    if success:
        print("\n✅ All checks completed successfully!")
        exit(0)
    else:
        print("\n❌ Some checks failed. See errors above.")
        exit(1)


if __name__ == "__main__":
    main()
