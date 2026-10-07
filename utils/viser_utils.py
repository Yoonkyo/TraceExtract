"""Record3D visualizer

Parse and stream record3d captures. To get the demo data, see `./assets/download_record3d_dance.sh`.
"""

import time
import sys

import numpy as np
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf
from matplotlib import colormaps
from loguru import logger
import torch

from models.SpaTrackV2.utils import visualizer as spatrack2_visualizer
from utils.threed_utils import (
    project_tracks_3d_to_2d,
    transform_points_to_coordinate,
)


def define_track_colors(pts, colormap='turbo'):
    """
    Determines colors for each point in a set of 2D points using a colormap.

    Parameters:
    - pts: List of points [(x, y, z), ...]
    - colormap: Name of the colormap to use

    Returns:
    - colors: List of colors for each point
    """
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    if np.all(maxs == mins):
        maxs = mins + 1
    pts_norm = (pts - mins) / (maxs - mins)
    orders = np.argsort(np.argsort([np.square(pt).sum() for pt in pts_norm])) / max((len(pts) - 1), 1)
    if len(orders) == 1:
        orders = np.array([0.5])
    return np.array([colormaps[colormap](order)[:3] for order in orders])


def overlay_tracks2d_on_video(video, tracks2d, track_colors=None):
    track_visualizer = spatrack2_visualizer.Visualizer()
    video_viz = track_visualizer.draw_tracks_on_video(
        torch.from_numpy(video).permute(0, 3, 1, 2)[None] * 255,
        torch.from_numpy(tracks2d)[None],
        visibility=None,
        track_colors=track_colors * 255.,
    )[0].permute(0, 2, 3, 1).numpy() / 255.
    return video_viz


def visualize(
    loader,
    downsample_factor: int = 1,
    max_frames: int = 100,
    share: bool = False,
    port: int = 8080,
) -> None:
    server = viser.ViserServer(port=port)
    if share:
        server.request_share_url()

    server.scene.set_up_direction("-y")
    logger.debug("[Viser] Loading frames!")
    length = min(max_frames, loader.length)

    is_playing = True
    visualized_video = None
    fixedview_video = None
    origin_coordinate_c2w = loader.get_cam(0)['c2w']
    camera_frustums = []

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_pause_btn = server.gui.add_button("Play/Pause")
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=length - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.001,
            max=0.02,
            step=1e-3,
            initial_value=0.005,
        )
        gui_track_size = server.gui.add_slider(
            "Track size",
            min=0.2,
            max=3,
            step=0.2,
            initial_value=1.5,
        )
        gui_track_range = server.gui.add_slider(
            "Track range",
            min=-1,
            max=100,
            step=1,
            initial_value=10,
        )
        gui_frustum_scale = server.gui.add_slider(
            "Frustum scale",
            min=0.01,
            max=0.3,
            step=0.01,
            initial_value=0.15,
        )
        gui_show_origin = server.gui.add_checkbox("World origin", True)
        gui_show_pcd = server.gui.add_checkbox("Point cloud", True)
        gui_show_track_est = server.gui.add_checkbox("3D tracks", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=10
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    with server.gui.add_folder("Video"):
        gui_video_handle = server.gui.add_image(loader.get_frame(0), label="Input video")
        gui_converted_video_handle = server.gui.add_image(loader.get_frame(0) * 0.5, label="Fixed-view video")
        gui_convert_frame_id = server.gui.add_slider("Convert frame", 0, length - 1, 1, initial_value=0)
        gui_convert_frame_btn = server.gui.add_button("Convert coordinate")

    # visualize video
    tracks3d, _, tracks_colors = loader.get_all_tracks()  # (T, N, 3), _, (N, 3)
    tracks2d = project_tracks_3d_to_2d(
        tracks3d=tracks3d,  # (T, N, 3)
        camera_views=[loader.get_cam(i) for i in range(length)],
    )  # (T, N, 2)

    visualized_video = overlay_tracks2d_on_video(
        loader.get_all_frames(),
        tracks2d,
        tracks_colors,
    )

    def _visualize_fixedview_video():
        nonlocal fixedview_video, tracks3d, tracks_colors
        tracks2d = project_tracks_3d_to_2d(
            tracks3d=tracks3d,  # (T, N, 3)
            camera_views=[loader.get_cam(gui_convert_frame_id.value)] * length,
        )  # (T, N, 2)
        fixedview_video = overlay_tracks2d_on_video(
            np.repeat(loader.get_frame(gui_convert_frame_id.value)[None], length, axis=0),
            tracks2d,
            tracks_colors,
        )

    def _transform_origin():
        nonlocal origin_node, origin_coordinate_c2w
        origin_points = origin_node.points.copy()  # (N, 2, 3)
        origin_points = origin_points.reshape(-1, 3)

        new_c2w_coordinate = loader.get_cam(gui_convert_frame_id.value)['c2w']
        origin_points = transform_points_to_coordinate(
            points=origin_points[None],
            Rt=(new_c2w_coordinate @ np.linalg.inv(origin_coordinate_c2w))[None],
        )[0]
        origin_node.points = origin_points.reshape(-1, 2, 3)
        origin_coordinate_c2w = new_c2w_coordinate

    def _extract_time_from_node(node) -> int:
        # Extract the time from the node name.
        node_name = node.name
        if node_name.endswith('/point_cloud'):
            node_name = node_name[:-12]
        elif node_name.endswith('/tracks'):
            node_name = node_name[:-7]
        t = int(node_name.split('/')[-1])
        return t

    @gui_pause_btn.on_click
    def _(_) -> None:
        nonlocal is_playing
        is_playing = not is_playing
        # disable buttons when playing
        gui_timestep.disabled = is_playing
        gui_next_frame.disabled = is_playing
        gui_prev_frame.disabled = is_playing

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % length

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % length

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    @gui_frustum_scale.on_update
    def _(_) -> None:
        with server.atomic():
            for frustum in camera_frustums:
                frustum.scale = gui_frustum_scale.value
        server.flush()

    @gui_show_origin.on_update
    def _(_) -> None:
        with server.atomic():
            origin_node.visible = gui_show_origin.value
        server.flush()

    @gui_convert_frame_btn.on_click
    def _(_) -> None:
        frame_id = gui_convert_frame_id.value
        _visualize_fixedview_video()
        _transform_origin()

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            # Toggle visibility.
            for frame_node in frame_nodes:
                t = _extract_time_from_node(frame_node)
                if t == current_timestep:
                    frame_node.visible = True
                else:
                    frame_node.visible = False
            for point_node in point_nodes:
                t = _extract_time_from_node(point_node)
                if t == current_timestep and gui_show_pcd.value:
                    point_node.visible = True
                else:
                    point_node.visible = False

            for track_node in track_nodes:
                t = _extract_time_from_node(track_node)
                if gui_track_range.value >= 0:
                    if t <= current_timestep and t >= current_timestep - gui_track_range.value:
                        if gui_show_track_est.value:
                            track_node.visible = True
                        else:
                            track_node.visible = False
                    else:
                        track_node.visible = False
                else:
                    track_node.visible = False

            # show video
            if visualized_video is not None:
                gui_video_handle.image = visualized_video[current_timestep]
            if fixedview_video is not None:
                gui_converted_video_handle.image = fixedview_video[current_timestep]

        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Load in frames.
    for name in ["/frames", "/tracks", "/pointcloud"]:
        server.scene.add_frame(
            name,
            position=(0, 0, 0),
            show_axes=False,
        )

    frame_nodes: list[viser.FrameHandle] = []
    point_nodes: list[viser.PointCloudHandle] = []
    track_nodes: list[viser.LineSegmentsHandle] = []

    origin_node = server.scene.add_line_segments(
        name="origin",
        points=np.array([
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]],
        ]),
        colors=np.array([
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ]),
        line_width=gui_track_size.value,
    )
    _transform_origin()
    _visualize_fixedview_video()

    for t_id in tqdm(range(length)):
        view_name = f"/frames/{t_id:03d}"
        point_name = f"/pointcloud/{t_id:03d}"
        track_name = f"/tracks/{t_id:03d}"

        frame = loader.get_frame(t_id)
        points_xyz = loader.get_point_cloud(t_id)
        points_xyz = points_xyz[::downsample_factor, ::downsample_factor].reshape(-1, 3)
        points_rgb = frame[::downsample_factor, ::downsample_factor].reshape(-1, 3)
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"{point_name}/point_cloud",
                points=points_xyz,
                colors=points_rgb,
                point_size=gui_point_size.value,
                point_shape="rounded",
            )
        )
        point_nodes[-1].visible = gui_show_pcd.value

        tracks_xyz_prev, _, _ = loader.get_tracks(max(t_id - 1, 0))
        tracks_xyz_curr, _, tracks_rgb = loader.get_tracks(t_id)
        segment_xyzs = np.stack([tracks_xyz_prev, tracks_xyz_curr], axis=1)
        segment_rgbs = np.stack([tracks_rgb, tracks_rgb], axis=1)
        track_nodes.append(
            server.scene.add_line_segments(
                name=f"{track_name}/tracks",
                points=segment_xyzs,
                colors=segment_rgbs,
                line_width=gui_track_size.value,
            )
        )

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(view_name, show_axes=False))

        cam_params = loader.get_cam(t_id)
        if cam_params is not None:
            c2w = cam_params['c2w']

            # Place the frustum.
            fov = 2 * np.arctan2(cam_params['height'] / 2, cam_params['K'][0, 0])
            aspect = cam_params['width'] / cam_params['height']
            frustum = server.scene.add_camera_frustum(
                f"{view_name}/frustum",
                fov=fov,
                aspect=aspect,
                scale=gui_frustum_scale.value,
                image=frame[::downsample_factor, ::downsample_factor],
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
            )
            camera_frustums.append(frustum)

        # Add some axes.
        server.scene.add_frame(
            f"{view_name}/frustum/axes",
            axes_length=0.05,
            axes_radius=0.005,
        )

    if not frame_nodes:
        raise ValueError("No frame nodes created.")
        sys.exit(1)

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        t = _extract_time_from_node(frame_node)
        frame_node.visible = t == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        # Update the timestep if we're playing.
        if is_playing:
            gui_timestep.value = (gui_timestep.value + 1) % length

        # Update point size of both this timestep and the next one! There's
        # redundancy here, but this will be optimized out internally by viser.
        #
        # We update the point size for the next timestep so that it will be
        # immediately available when we toggle the visibility.
        for node in point_nodes:
            t = _extract_time_from_node(node)
            if t in [gui_timestep.value, (gui_timestep.value + 1) % length]:
                node.point_size = gui_point_size.value

        for node in track_nodes:
            t = _extract_time_from_node(node)
            if gui_track_range.value >= 0:
                if t <= gui_timestep.value and t >= gui_timestep.value - gui_track_range.value:
                    node.line_width = gui_track_size.value

        time.sleep(1.0 / gui_framerate.value)
