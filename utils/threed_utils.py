import os
import sys
import numpy as np
import torch


def inverse_intrinsic(K):
    K_inv = np.eye(3)[None].repeat(len(K), axis=0)
    K_inv[:, :2, -1] = K[:, :2, -1] * -1
    K_inv[:, :1, :] /= K[:, :1, :1]
    K_inv[:, 1:2, :] /= K[:, 1:2, 1:2]
    return K_inv


def get_meshgrid(h, w, homogeneous=False, pixel_center=False):
    '''
    Make a meshgrid. meshgrid[y, x] = (x, y)
    Input:
        h: height
        w: width
        homogeneous: default False
    Return:
        meshgrid: np.array [1, 2, H, W] if not homogeneous else [1, 3, H, W]
    '''
    if pixel_center:
        offset = 0.5
    else:
        offset = 0.0
    x = np.arange(0, w) + offset
    y = np.arange(0, h) + offset
    yv, xv = np.meshgrid(y, x, indexing='ij')
    to_stack = [xv, yv]
    if homogeneous:
        to_stack += [np.ones_like(xv)]
    meshgrid = np.stack(to_stack, 0)
    return meshgrid[None]


def unproject_by_depth(depth, K, c2w=None):
    '''
    Unproject depth map to 3D points.
    Input:
        depth: np.array [B, 1, H, W]
        K: np.array [B, 3, 3]
        c2w: np.array [B, 4, 4] or None
    Return:
        xyz: np.array [B, 3, H, W]
    '''
    B, _, H, W = depth.shape

    xyz = get_meshgrid(H, W, homogeneous=True).repeat(B, axis=0).reshape(B, 3, -1)
    xyz = (inverse_intrinsic(K) @ xyz) * depth.reshape(B, 1, H * W)
    if c2w is not None:
        xyz = c2w[:, :3, :3] @ xyz + c2w[:, :3, 3:]
    return xyz.reshape(B, 3, H, W)


def transform_points_to_coordinate(points, Rt):
    """
    Transform points to a new coordinate system defined by Rt.
    Input:
        points: np.array [B, N, 3]
        Rt: np.array [B, 4, 4]
    Return:
        points_transformed: np.array [B, N, 3]
    """
    R = Rt[:, :3, :3]
    t = Rt[:, :3, 3:]
    points_transformed = (R @ points.transpose(0, 2, 1)) + t
    return points_transformed.transpose(0, 2, 1)  # (B, N, 3)


def project_tracks_3d_to_2d(tracks3d, camera_views):
    """
    Input:
        tracks3d: np.array [T, N, 3]
        camera_views: list of dict, len T, each dict has keys:
            'K': np.array [3, 3]
            'c2w': np.array [4, 4]
    Return:
        tracks2d: np.array [T, N, 2]
    """
    assert len(tracks3d) == len(camera_views)

    c2ws = np.stack([cv['c2w'] for cv in camera_views], 0)  # (T, 4, 4)
    w2cs = np.linalg.inv(c2ws)  # (T, 4, 4)
    Ks = np.stack([cv['K'] for cv in camera_views], 0)  # (T, 3, 3)

    T, N, _ = tracks3d.shape
    tracks_camcoord = transform_points_to_coordinate(tracks3d, w2cs)
    tracks_imgcoord = Ks @ tracks_camcoord.transpose(0, 2, 1)  # (T, 3, N)
    tracks_imgcoord = tracks_imgcoord[:, :2, :] / (tracks_imgcoord[:, 2:3, :] + 1e-8)  # (T, 2, N)
    # in this point, x is between 0 and W, y is between 0 and H, Z is some value without scale. 
    # I should use this Z (match scale between xy and z)
    # tracks_imgcoord[:,0] is between 0 and W, tracks_imgcoord[:10] is between 0 and H, Z is some value without scale. 
    # tracks_imgcoord[:,0] - W/2 / W/2, tracks_imgcoord[:10] - H/2 / H/2, Z is some value without scale. 
    # if I get output of NN, then I can do tracks_imgcoord[:,0] * W/2 + W/2, tracks_imgcoord[:10] * H/2 + H/2, tracks_imgcoord[:,3] * 1
    # -> then, if I have K, then I should be able to get the world coordinate of the points.    
    return tracks_imgcoord.transpose(0, 2, 1)  # (T, N, 2)


def project_tracks_3d_to_3d(tracks3d, camera_views):
    """
    Input:
        tracks3d: np.array [T, N, 3]
        camera_views: list of dict, len T, each dict has keys:
            'K': np.array [3, 3]
            'c2w': np.array [4, 4]
    Return:
        tracks2d: np.array [T, N, 2]
    """
    assert len(tracks3d) == len(camera_views)

    c2ws = np.stack([cv['c2w'] for cv in camera_views], 0)  # (T, 4, 4)
    w2cs = np.linalg.inv(c2ws)  # (T, 4, 4)
    Ks = np.stack([cv['K'] for cv in camera_views], 0)  # (T, 3, 3)

    T, N, _ = tracks3d.shape
    tracks_camcoord = transform_points_to_coordinate(tracks3d, w2cs)
    tracks_imgcoord = Ks @ tracks_camcoord.transpose(0, 2, 1)  # (T, 3, N)
    tracks_imgcoord_return = tracks_imgcoord[:, :2, :] / (tracks_imgcoord[:, 2:3, :] + 1e-8)  # (T, 2, N)
    tracks_imgcoord_return = np.concatenate([tracks_imgcoord_return, tracks_imgcoord[:, 2:3, :]], axis=1)  # (T, 3, N)
    # in this point, x is between 0 and W, y is between 0 and H, Z is some value without scale. 
    # I should use this Z (match scale between xy and z)
    # tracks_imgcoord[:,0] is between 0 and W, tracks_imgcoord[:10] is between 0 and H, Z is some value without scale. 
    # tracks_imgcoord[:,0] - W/2 / W/2, tracks_imgcoord[:10] - H/2 / H/2, Z is some value without scale. 
    # if I get output of NN, then I can do tracks_imgcoord[:,0] * W/2 + W/2, tracks_imgcoord[:10] * H/2 + H/2, tracks_imgcoord[:,3] * 1
    # -> then, if I have K, then I should be able to get the world coordinate of the points.    
    return tracks_imgcoord_return.transpose(0, 2, 1)  # (T, N, 2)
