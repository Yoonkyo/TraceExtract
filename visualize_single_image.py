import os
import sys
import argparse
import numpy as np
import cv2 as cv
from PIL import Image
import viser
import viser.extras
import viser.transforms as tf
from utils.viser_utils import define_track_colors
from utils.threed_utils import unproject_by_depth, inverse_intrinsic, get_meshgrid
from utils.output_format import load_image, load_depth, load_sample, is_unified_format, get_image_shape
from loguru import logger


"""
Usage (new unified .npy format):
    python visualize_single_image.py \
        --video_dir <output_dir>/<video_name> \
        --frame_index <frame> \
        --port 8080

Usage (legacy per-frame format):
    python visualize_single_image.py \
        --npz_path <output_dir>/<video_name>/samples/<video_name>_<frame>.npz \
        --image_path <output_dir>/<video_name>/images/<video_name>_<frame>.png \
        --depth_path <output_dir>/<video_name>/depth/<video_name>_<frame>.png \
        --port 8080
"""

def load_depth_from_path(depth_path):
    """Load depth data from either PNG or NPZ file"""
    if depth_path.endswith(".npz"):
        # Load raw depth from NPZ
        depth_data = np.load(depth_path)
        depth = depth_data["depth"]
        depth_data.close()
        return depth
    elif depth_path.endswith(".png"):
        # Check if there's a corresponding raw NPZ file
        base_path = depth_path[:-4]  # Remove .png extension
        raw_npz_path = f"{base_path}_raw.npz"
        if os.path.exists(raw_npz_path):
            depth_data = np.load(raw_npz_path)
            depth = depth_data["depth"]
            depth_data.close()
            return depth
        else:
            # Load from PNG (16-bit, need to convert back from mm)
            depth_img = np.array(Image.open(depth_path))
            depth = depth_img.astype(np.float32) / 10000.0  # Convert back from mm
            return depth
    else:
        raise ValueError(f"Unsupported depth file format: {depth_path}")


def get_camera_params_from_main_npz(episode_dir, frame_idx):
    """Get camera intrinsics and extrinsics for a single frame.

    Prefers the lightweight ``cameras.npz`` (intrinsics + extrinsics + image
    size only), falling back to the legacy per-video ``<name>.npz`` bundle
    when present.
    """
    episode_name = os.path.basename(episode_dir)
    cameras_path = os.path.join(episode_dir, "cameras.npz")
    main_npz_path = os.path.join(episode_dir, f"{episode_name}.npz")

    if os.path.exists(cameras_path):
        data = np.load(cameras_path)
        intrinsics = data["intrinsics"][frame_idx]  # (3, 3)
        extrinsics = data["extrinsics"][frame_idx]  # (4, 4) - world to camera
        c2w = np.linalg.inv(extrinsics)
        height = int(data["height"]) if "height" in data.files else 0
        width = int(data["width"]) if "width" in data.files else 0
        data.close()
        if height == 0 or width == 0:
            # Older cameras.npz didn't store image size — peek at saved data.
            _, height, width = get_image_shape(episode_dir, episode_name)

    elif os.path.exists(main_npz_path):
        data = np.load(main_npz_path)
        intrinsics = data["intrinsics"][frame_idx]
        extrinsics = data["extrinsics"][frame_idx]
        c2w = np.linalg.inv(extrinsics)
        height, width = int(data["height"]), int(data["width"])
        data.close()

    else:
        print(f"Camera params not found at {cameras_path} or {main_npz_path}")
        # Use hardcoded camera parameters
        intrinsics = np.array([
            [257.91296, 0.0, 259.0],
            [0.0, 261.4576, 161.0],
            [0.0, 0.0, 1.0]
        ])
        
        extrinsics = np.array([
            [1.0000000e+00, 4.0706014e-05, 8.9567264e-05, 9.0881156e-05],
            [-4.0680898e-05, 9.9999994e-01, -2.8039535e-04, 3.5203320e-05],
            [-8.9578680e-05, 2.8039169e-04, 9.9999994e-01, -2.7687754e-04],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        c2w = np.linalg.inv(extrinsics)  # camera to world
        height, width = 322, 518

    return {
        "K": intrinsics,
        "c2w": c2w,
        "w2c": extrinsics,
        "height": height,
        "width": width,
    }


def convert_image_coords_to_world(traj_image_coords, camera_params):
    """
    Convert trajectories from image coordinates (x,y,z) to world coordinates.

    Any input row containing -inf or NaN (the per-track padding sentinel) is
    preserved as -inf in the output rather than being passed through the
    camera matrices, which would otherwise yield NaN/inf world coordinates.

    Args:
        traj_image_coords: (N, H, 3) trajectories in image coordinates
        camera_params: dict with camera parameters

    Returns:
        traj_world: (N, H, 3) trajectories in world coordinates
    """
    N, H, _ = traj_image_coords.shape
    K = camera_params["K"]
    c2w = camera_params["c2w"]

    traj_flat = traj_image_coords.reshape(N * H, 3)  # (N*H, 3)
    invalid_mask = ~np.isfinite(traj_flat).all(axis=1)  # (N*H,)

    # Replace invalid rows with zeros so the arithmetic below stays finite;
    # we will overwrite these rows with -inf at the end.
    safe_flat = traj_flat.copy()
    safe_flat[invalid_mask] = 0.0

    world_points = np.empty((N * H, 3), dtype=traj_image_coords.dtype)
    for i in range(N * H):
        x, y, z = safe_flat[i]

        # Convert pixel coordinates to normalized camera coordinates
        x_norm = (x - K[0, 2]) / K[0, 0]
        y_norm = (y - K[1, 2]) / K[1, 1]

        # Create point in camera coordinates
        cam_point = np.array([x_norm * z, y_norm * z, z, 1.0])

        # Transform to world coordinates
        world_point = c2w @ cam_point
        world_points[i] = world_point[:3]

    world_points[invalid_mask] = -np.inf
    return world_points.reshape(N, H, 3)


def visualize_single_image(npz_path=None, image_path=None, depth_path=None,
                           video_dir=None, frame_index=None,
                           port=8080, visualize_history=False, use_raw=False):
    """Visualize 3D scene with trajectories for a single image.

    Accepts either explicit file paths (legacy) or ``video_dir`` +
    ``frame_index`` which auto-detects the on-disk format.
    """

    # ----- Resolve episode_dir and frame_idx from the two calling modes ------
    if video_dir is not None and frame_index is not None:
        episode_dir = video_dir
        frame_idx = frame_index
        episode_name = os.path.basename(episode_dir)
    elif npz_path is not None:
        sample_dir = os.path.dirname(npz_path)
        episode_dir = os.path.dirname(sample_dir)
        npz_filename = os.path.basename(npz_path)
        frame_idx = int(npz_filename.split("_")[-1].split(".")[0])
        episode_name = os.path.basename(episode_dir)
    else:
        raise ValueError("Provide either --video_dir + --frame_index or --npz_path")

    logger.info(f"Loading data for frame {frame_idx} from {episode_dir}")

    # ----- Load sample data (auto-detects format) ----------------------------
    sample_data = load_sample(episode_dir, episode_name, frame_idx)

    if use_raw:
        if visualize_history and "raw_traj_history" in sample_data:
            traj_image_coords = sample_data["raw_traj_history"]
            valid_steps_raw = sample_data.get(
                "raw_valid_steps_history",
                sample_data.get("valid_steps_history", sample_data["valid_steps"]))
            logger.info("Visualizing raw_traj_history (past, pre-retarget)")
        else:
            traj_image_coords = sample_data["raw_traj"]
            valid_steps_raw = sample_data.get("raw_valid_steps", sample_data["valid_steps"])
            logger.info("Visualizing raw_traj (future, pre-retarget)")
    else:
        if visualize_history and "traj_history" in sample_data:
            traj_image_coords = sample_data["traj_history"]
            valid_steps_raw = sample_data.get("valid_steps_history", sample_data["valid_steps"])
            logger.info("Visualizing traj_history (past trajectories)")
        else:
            traj_image_coords = sample_data["traj"]
            valid_steps_raw = sample_data["valid_steps"]
    keypoints = sample_data["keypoints"]
    is_moving = sample_data.get("is_moving")

    # Per-track masking: mask now has shape (N, H). Older files may still
    # carry a 1D (H,) mask — broadcast it so each track gets its own row.
    valid_steps_arr = np.asarray(valid_steps_raw)
    if valid_steps_arr.ndim == 1:
        valid_steps_arr = np.broadcast_to(
            valid_steps_arr.astype(bool),
            (traj_image_coords.shape[0], traj_image_coords.shape[1]),
        ).copy()
    else:
        valid_steps_arr = valid_steps_arr.astype(bool)

    # Filter to moving keypoints only (if available)
    if is_moving is not None:
        moving_mask = is_moving.astype(bool)
        traj_image_coords = traj_image_coords[moving_mask]
        keypoints = keypoints[moving_mask]
        valid_steps_arr = valid_steps_arr[moving_mask]
        logger.info(
            f"Filtered to {len(traj_image_coords)} moving keypoints "
            f"(out of {len(moving_mask)} total)"
        )

    logger.info(
        f"Loaded {len(traj_image_coords)} trajectories with horizon {traj_image_coords.shape[1]}"
    )

    # Load RGB image (auto-detect format)
    if image_path is not None and os.path.isfile(image_path):
        image = np.array(Image.open(image_path)).astype(np.float32) / 255.0
    else:
        image = load_image(episode_dir, episode_name, frame_idx).astype(np.float32) / 255.0
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)

    # Load depth data (auto-detect format)
    if depth_path is not None and os.path.isfile(depth_path):
        depth = load_depth_from_path(depth_path)
    else:
        depth = load_depth(episode_dir, episode_name, frame_idx)

    logger.info(f"Image shape: {image.shape}, Depth shape: {depth.shape}")

    # Get camera parameters
    camera_params = get_camera_params_from_main_npz(episode_dir, frame_idx)

    # Convert trajectories from image coordinates to world coordinates
    traj_world = convert_image_coords_to_world(traj_image_coords, camera_params)

    # Create point cloud from RGB image and depth
    H, W = depth.shape
    points_xyz = unproject_by_depth(
        depth=depth[None, None],  # (1, 1, H, W)
        K=camera_params["K"][None],  # (1, 3, 3)
        c2w=camera_params["c2w"][None],  # (1, 4, 4)
    )[0].transpose(1, 2, 0)  # (H, W, 3)

    # Downsample point cloud for visualization
    downsample_factor = 4
    points_xyz_ds = points_xyz[::downsample_factor, ::downsample_factor].reshape(-1, 3)
    points_rgb_ds = image[::downsample_factor, ::downsample_factor].reshape(-1, 3)

    # Filter out invalid points (depth = 0 or too far)
    valid_mask = (points_xyz_ds[:, 2] > 0) & (
        points_xyz_ds[:, 2] < 10.0
    )  # Filter points within 10m
    points_xyz_ds = points_xyz_ds[valid_mask]
    points_rgb_ds = points_rgb_ds[valid_mask]

    logger.info(f"Point cloud: {len(points_xyz_ds)} points after filtering")

    # Define colors for trajectories
    track_colors = define_track_colors(traj_world, colormap='turbo')

    # Start Viser server
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("-y")

    logger.info(f"Started Viser server at http://localhost:{port}")

    # Add GUI controls
    with server.gui.add_folder("Visualization"):
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.02, step=1e-3, initial_value=0.006
        )
        gui_track_width = server.gui.add_slider(
            "Track width", min=0.5, max=5.0, step=0.5, initial_value=4.0
        )
        gui_track_length = server.gui.add_slider(
            "Track length",
            min=1,
            max=traj_world.shape[1],
            step=1,
            initial_value=min(30, traj_world.shape[1]),
        )
        gui_show_pointcloud = server.gui.add_checkbox("Show point cloud", True)
        gui_show_tracks = server.gui.add_checkbox("Show tracks", True)
        gui_show_keypoints = server.gui.add_checkbox("Show keypoints", False)
        gui_keypoint_size = server.gui.add_slider(
            "Keypoint size", min=0.005, max=0.05, step=0.005, initial_value=0.005
        )
        gui_show_frustum = server.gui.add_checkbox("Show camera frustum", True)
        gui_show_axes = server.gui.add_checkbox("Show world axes", True)

    # Add point cloud
    point_cloud_handle = server.scene.add_point_cloud(
        name="point_cloud",
        points=points_xyz_ds,
        colors=points_rgb_ds,
        point_size=gui_point_size.value,
        point_shape="rounded",
    )

    # Add trajectories as line segments
    track_handles = []
    keypoint_handles = []

    for i, (traj, color) in enumerate(zip(traj_world, track_colors)):
        # Slice to current track length, then keep only finite & valid steps
        L = gui_track_length.value
        slice_traj = traj[:L]
        slice_mask = valid_steps_arr[i, :L] & np.isfinite(slice_traj).all(axis=1)
        valid_traj = slice_traj[slice_mask]
        if len(valid_traj) > 1:
            segments = []
            seg_colors = []
            for j in range(len(valid_traj) - 1):
                segments.append([valid_traj[j], valid_traj[j + 1]])
                seg_colors.append([color, color])

            if segments:
                track_handle = server.scene.add_line_segments(
                    name=f"track_{i}",
                    points=np.array(segments),
                    colors=np.array(seg_colors),
                    line_width=gui_track_width.value,
                )
                track_handles.append(track_handle)

        # Add starting keypoint in 3D (convert from image coordinates to world)
        kp_x, kp_y = keypoints[i]
        kp_depth = (
            depth[int(kp_y), int(kp_x)] if 0 <= kp_x < W and 0 <= kp_y < H else 1.0
        )

        # Convert keypoint to world coordinates
        kp_world = convert_image_coords_to_world(
            np.array([[[kp_x, kp_y, kp_depth]]]), camera_params
        )[0, 0]

        keypoint_handle = server.scene.add_point_cloud(
            name=f"keypoint_{i}",
            points=kp_world[None],
            colors=color[None],
            point_size=gui_keypoint_size.value,
            point_shape="circle",
        )
        keypoint_handle.visible = False  # Initially hidden
        keypoint_handles.append(keypoint_handle)

    # Add camera frame
    c2w = camera_params["c2w"]
    fov = 2 * np.arctan2(camera_params["height"] / 2, camera_params["K"][0, 0])
    aspect = camera_params["width"] / camera_params["height"]

    frustum_handle = server.scene.add_camera_frustum(
        name="camera_frustum",
        fov=fov,
        aspect=aspect,
        scale=0.1,
        image=image,
        wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
        position=c2w[:3, 3],
    )
    frustum_handle.visible = gui_show_frustum.value 

    # Add coordinate axes at world origin
    axes_handle = server.scene.add_line_segments(
        name="world_axes",
        points=np.array([
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
        ]),
        colors=np.array([
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ]),
        line_width=3.0,
    )
    axes_handle.visible = gui_show_axes.value

    # Update callbacks for GUI controls
    @gui_point_size.on_update
    def _(_) -> None:
        if gui_show_pointcloud.value:
            point_cloud_handle.point_size = gui_point_size.value

    @gui_track_width.on_update
    def _(_) -> None:
        if gui_show_tracks.value:
            for handle in track_handles:
                handle.line_width = gui_track_width.value

    @gui_keypoint_size.on_update
    def _(_) -> None:
        if gui_show_keypoints.value:
            for handle in keypoint_handles:
                handle.point_size = gui_keypoint_size.value

    @gui_show_pointcloud.on_update
    def _(_) -> None:
        point_cloud_handle.visible = gui_show_pointcloud.value

    @gui_show_tracks.on_update
    def _(_) -> None:
        for handle in track_handles:
            handle.visible = gui_show_tracks.value

    @gui_show_keypoints.on_update
    def _(_) -> None:
        for handle in keypoint_handles:
            handle.visible = gui_show_keypoints.value

    @gui_show_frustum.on_update
    def _(_ev):
        frustum_handle.visible = gui_show_frustum.value

    @gui_show_axes.on_update
    def _(_ev):
        axes_handle.visible = gui_show_axes.value

    @gui_track_length.on_update
    def _(_) -> None:
        # Remove old track handles
        for handle in track_handles:
            handle.remove()
        track_handles.clear()

        # Create new tracks with updated length
        for i, (traj, color) in enumerate(zip(traj_world, track_colors)):
            L = gui_track_length.value
            slice_traj = traj[:L]
            slice_mask = valid_steps_arr[i, :L] & np.isfinite(slice_traj).all(axis=1)
            valid_traj = slice_traj[slice_mask]
            if len(valid_traj) > 1:
                segments = []
                seg_colors = []
                for j in range(len(valid_traj) - 1):
                    segments.append([valid_traj[j], valid_traj[j + 1]])
                    seg_colors.append([color, color])

                if segments:
                    track_handle = server.scene.add_line_segments(
                        name=f"track_{i}_updated",
                        points=np.array(segments),
                        colors=np.array(seg_colors),
                        line_width=gui_track_width.value,
                    )
                    track_handle.visible = gui_show_tracks.value
                    track_handles.append(track_handle)

    logger.info("Visualization ready! Press Ctrl+C to exit.")

    # Keep the server running
    try:
        while True:
            import time

            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize 3D scene with trajectories for a single image"
    )
    # New unified format
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Video output directory (new .npy format). Use with --frame_index.",
    )
    parser.add_argument(
        "--frame_index", type=int, default=None,
        help="Frame index to visualize (used with --video_dir).",
    )
    # Legacy per-file format
    parser.add_argument(
        "--npz_path", type=str, default=None,
        help="Path to per-frame NPZ sample (legacy format)",
    )
    parser.add_argument(
        "--image_path", type=str, default=None,
        help="Path to the RGB image (legacy format)",
    )
    parser.add_argument(
        "--depth_path", type=str, default=None,
        help="Path to the depth image/data (legacy format)",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for Viser server (default: 8080)"
    )
    parser.add_argument(
        "--visualize_history", action="store_true",
        help="Visualize traj_history (past trajectories) instead of traj (future)"
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Visualize raw_traj/raw_traj_history (pre-retarget, frame-aligned) "
             "instead of the arc-length retargeted traj."
    )

    args = parser.parse_args()

    if args.video_dir is not None:
        if args.frame_index is None:
            parser.error("--frame_index is required when using --video_dir")
        visualize_single_image(
            video_dir=args.video_dir, frame_index=args.frame_index,
            port=args.port, visualize_history=args.visualize_history, use_raw=args.raw,
        )
    elif args.npz_path is not None:
        if not os.path.exists(args.npz_path):
            raise FileNotFoundError(f"NPZ file not found: {args.npz_path}")
        visualize_single_image(
            npz_path=args.npz_path, image_path=args.image_path,
            depth_path=args.depth_path, port=args.port,
            visualize_history=args.visualize_history, use_raw=args.raw,
        )
    else:
        parser.error("Provide either --video_dir + --frame_index or --npz_path")
