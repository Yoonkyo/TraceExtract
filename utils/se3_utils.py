"""SE3 utilities for coordinate alignment across VGGT chunks.

Provides quaternion operations, SE3 averaging, and least-squares
alignment of dense chunks to a global sparse reference frame.
"""

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
#  Quaternion utilities  (convention: [w, x, y, z])
# ---------------------------------------------------------------------------

def quaternion_from_matrix(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a unit quaternion [w, x, y, z].

    Uses Shepperd's method for numerical stability.
    """
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion [w, x, y, z] to a 3×3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def mean_quaternion(quaternions: np.ndarray) -> np.ndarray:
    """Compute the mean of multiple unit quaternions [w, x, y, z].

    Uses the eigendecomposition of the quaternion outer-product matrix
    (Markley et al. 2007). The eigenvector corresponding to the largest
    eigenvalue is the mean quaternion.

    Args:
        quaternions: (N, 4) array of unit quaternions.

    Returns:
        (4,) mean quaternion [w, x, y, z].
    """
    Q = np.asarray(quaternions, dtype=np.float64)
    assert Q.ndim == 2 and Q.shape[1] == 4

    # Ensure consistent hemisphere (all quaternions on the same side of the
    # hypersphere as the first one) to avoid averaging antipodal pairs.
    for i in range(1, len(Q)):
        if np.dot(Q[i], Q[0]) < 0:
            Q[i] = -Q[i]

    # Outer-product matrix M = sum(q_i @ q_i^T)
    M = Q.T @ Q  # (4, 4)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # Largest eigenvalue is last (eigh returns ascending order)
    mean_q = eigenvectors[:, -1]
    mean_q /= np.linalg.norm(mean_q)
    # Ensure w >= 0 by convention
    if mean_q[0] < 0:
        mean_q = -mean_q
    return mean_q


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions.

    Args:
        q0, q1: (4,) unit quaternions [w, x, y, z].
        t: interpolation parameter in [0, 1].

    Returns:
        (4,) interpolated quaternion.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.dot(q0, q1)

    # Ensure shortest path
    if dot < 0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        # Very close — use linear interpolation for stability
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w0 = np.sin((1 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta
    result = w0 * q0 + w1 * q1
    return result / np.linalg.norm(result)


# ---------------------------------------------------------------------------
#  SE3 averaging and alignment
# ---------------------------------------------------------------------------

def compute_pairwise_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute the SE3 transform T such that target ≈ T @ source.

    Args:
        source: (4, 4) SE3 matrix.
        target: (4, 4) SE3 matrix.

    Returns:
        (4, 4) SE3 matrix T = target @ inv(source).
    """
    return target @ np.linalg.inv(source)


def mean_SE3(transforms: np.ndarray) -> np.ndarray:
    """Average multiple SE3 transforms.

    Rotation: quaternion mean via eigendecomposition.
    Translation: arithmetic mean.

    Args:
        transforms: (N, 4, 4) array of SE3 matrices.

    Returns:
        (4, 4) mean SE3 transform.
    """
    transforms = np.asarray(transforms, dtype=np.float64)
    N = transforms.shape[0]

    # Extract rotations and translations
    quaternions = np.array([quaternion_from_matrix(T[:3, :3]) for T in transforms])
    translations = transforms[:, :3, 3]

    mean_q = mean_quaternion(quaternions)
    mean_R = matrix_from_quaternion(mean_q)
    mean_t = translations.mean(axis=0)

    T_mean = np.eye(4, dtype=np.float64)
    T_mean[:3, :3] = mean_R
    T_mean[:3, 3] = mean_t
    return T_mean


def least_squares_SE3(
    sources: np.ndarray,
    targets: np.ndarray,
    outlier_sigma: float = 2.0,
) -> np.ndarray:
    """Compute the least-squares SE3 transform aligning sources to targets.

    Finds T such that targets[i] ≈ T @ sources[i] for all i,
    with optional outlier rejection.

    Args:
        sources: (N, 4, 4) SE3 matrices in the source (chunk-local) frame.
        targets: (N, 4, 4) SE3 matrices in the target (global) frame.
        outlier_sigma: reject anchor pairs whose alignment residual exceeds
            this many standard deviations. Set to 0 to disable.

    Returns:
        (4, 4) SE3 alignment transform.
    """
    sources = np.asarray(sources, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    assert sources.shape == targets.shape
    N = sources.shape[0]

    if N == 0:
        logger.warning("No anchor pairs for SE3 alignment — returning identity")
        return np.eye(4, dtype=np.float64)

    if N == 1:
        return compute_pairwise_transform(sources[0], targets[0])

    # Compute per-pair transforms
    pairwise = np.array([
        compute_pairwise_transform(sources[i], targets[i])
        for i in range(N)
    ])

    # Optional outlier rejection
    if outlier_sigma > 0 and N >= 4:
        # Compute initial mean
        T_init = mean_SE3(pairwise)

        # Compute residuals (translation distance after alignment)
        residuals = np.array([
            np.linalg.norm(
                (targets[i] @ np.linalg.inv(T_init @ sources[i]))[:3, 3]
            )
            for i in range(N)
        ])
        mean_res = residuals.mean()
        std_res = residuals.std()

        if std_res > 1e-10:
            inlier_mask = residuals < mean_res + outlier_sigma * std_res
            n_outliers = N - inlier_mask.sum()
            if n_outliers > 0:
                logger.info(
                    f"SE3 alignment: rejected {n_outliers}/{N} outlier anchor(s) "
                    f"(residual threshold: {mean_res + outlier_sigma * std_res:.4f})"
                )
                pairwise = pairwise[inlier_mask]

    return mean_SE3(pairwise)


def align_chunk_to_global(
    chunk_extrs: np.ndarray,
    sparse_extrs: np.ndarray,
    anchor_pairs: list,
    outlier_sigma: float = 2.0,
) -> np.ndarray:
    """Align a dense chunk's extrinsics to the global frame using anchor pairs.

    Args:
        chunk_extrs: (T_chunk, 4, 4) camera-to-world in chunk-local frame.
        sparse_extrs: (T_sparse, 4, 4) camera-to-world in global frame.
        anchor_pairs: list of (chunk_local_idx, sparse_idx) tuples identifying
            frames present in both the chunk and the sparse global set.
        outlier_sigma: outlier rejection threshold (σ). 0 to disable.

    Returns:
        (T_chunk, 4, 4) camera-to-world aligned to the global frame.
    """
    chunk_extrs = np.asarray(chunk_extrs, dtype=np.float64)
    sparse_extrs = np.asarray(sparse_extrs, dtype=np.float64)

    if len(anchor_pairs) == 0:
        logger.warning("No anchor pairs — returning chunk extrinsics unaligned")
        return chunk_extrs

    sources = np.array([chunk_extrs[local_idx] for local_idx, _ in anchor_pairs])
    targets = np.array([sparse_extrs[sparse_idx] for _, sparse_idx in anchor_pairs])

    T_align = least_squares_SE3(sources, targets, outlier_sigma=outlier_sigma)

    # Apply alignment to all frames in the chunk
    aligned = np.array([T_align @ chunk_extrs[i] for i in range(len(chunk_extrs))])

    logger.debug(
        f"Aligned chunk ({len(chunk_extrs)} frames) using "
        f"{len(anchor_pairs)} anchor(s)"
    )
    return aligned
