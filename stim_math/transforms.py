"""
Transformation matrices for three-phase electrode systems.

This module provides the mathematical transformations between different
coordinate systems used in three-phase electrostimulation:
- Alpha/Beta coordinates (2D position on phase diagram)
- Electrode potentials (N, L, R electrodes)
- Channel outputs (Left and Right audio channels)
"""

import numpy as np

# Electrode-to-channel projection matrix
# Maps electrode potentials [N, L, R] to channel outputs [Left, Right, Common]
# Row 1: Left channel = N - L
# Row 2: Right channel = N - R
# Row 3: Common (sum of all potentials, normally 0)
potential_to_channel_matrix = np.array([
    [1, -1, 0],
    [1, 0, -1],
    [1, 1, 1],
]).astype(np.float32)

potential_to_channel_matrix_inv = np.linalg.inv(potential_to_channel_matrix).astype(np.float32)

# Alpha-Beta transformation matrix
# Maps alpha/beta position to electrode potentials
# N electrode at top (0 degrees)
# L electrode at bottom-left (120 degrees)
# R electrode at bottom-right (240 degrees)
ab_transform = np.array([
    [1, 0, 1],                          # N: alpha component
    [-0.5, np.sqrt(3)/2, 1],            # L: 120 degree offset
    [-0.5, -np.sqrt(3)/2, 1]            # R: 240 degree offset
]).astype(np.float32)

ab_transform_inv = np.linalg.inv(ab_transform).astype(np.float32)

# Electrode vectors (for visualization and amplitude calculations)
n_vec = ab_transform[0, :2]  # [1, 0]
l_vec = ab_transform[1, :2]  # [-0.5, sqrt(3)/2]
r_vec = ab_transform[2, :2]  # [-0.5, -sqrt(3)/2]


def item_pos_to_ab(x: float, y: float) -> tuple:
    """
    Convert widget/pixel position to alpha/beta coordinates.

    Args:
        x: X position in widget (pixels)
        y: Y position in widget (pixels)

    Returns:
        (alpha, beta) tuple, values in range -1 to 1

    Note:
        Assumes a 166x166 pixel widget centered at (83, 83)
        where the unit circle has radius 83 pixels.
    """
    return y / -83.0, x / -83.0


def ab_to_item_pos(a: float, b: float) -> tuple:
    """
    Convert alpha/beta coordinates to widget/pixel position.

    Args:
        a: Alpha coordinate (-1 to 1)
        b: Beta coordinate (-1 to 1)

    Returns:
        (x, y) pixel position tuple

    Note:
        Returns position relative to widget center.
        Actual pixel position = center + returned value
    """
    return b * -83.0, a * -83.0


def half_angle_to_full(a: float, b: float) -> tuple:
    """
    Convert half angle (one dot) notation to full angle (two dots on opposite side).

    Used for converting between different electrode representation conventions.

    Args:
        a: Alpha coordinate
        b: Beta coordinate

    Returns:
        (a, b) converted coordinates
    """
    theta = np.arctan2(b, a)
    r = np.sqrt(a**2 + b**2)
    theta = theta / 2
    return np.cos(theta) * r, np.sin(theta) * r


def full_angle_to_half(a: float, b: float) -> tuple:
    """
    Convert full angle notation to half angle.

    Inverse of half_angle_to_full.

    Args:
        a: Alpha coordinate
        b: Beta coordinate

    Returns:
        (a, b) converted coordinates
    """
    theta = np.arctan2(b, a)
    r = np.sqrt(a**2 + b**2)
    theta = theta * 2
    return np.cos(theta) * r, np.sin(theta) * r


def ab_to_e123(a: float, b: float) -> np.ndarray:
    """
    Convert alpha/beta coordinates to electrode intensities (E1, E2, E3).

    Args:
        a: Alpha coordinate (-1 to 1)
        b: Beta coordinate (-1 to 1)

    Returns:
        numpy array [e1, e2, e3] with electrode intensities (0 to 1)
    """
    a, b = half_angle_to_full(a, b)

    # Apply ab transform
    e1, e2, e3 = ab_transform[:, :2] @ np.array([a, b])
    e = np.array([e1, e2, e3])

    # Ensure one component = 0 (normalize)
    min_val = np.min(np.abs(e))
    e = e - np.sign(e) * min_val
    e = e / 1.5
    e = np.abs(e)
    return e


def e123_to_ab(e1: float, e2: float, e3: float) -> tuple:
    """
    Convert electrode intensities to alpha/beta coordinates.

    Inverse of ab_to_e123.

    Args:
        e1: Electrode 1 intensity (0 to 1)
        e2: Electrode 2 intensity (0 to 1)
        e3: Electrode 3 intensity (0 to 1)

    Returns:
        (alpha, beta) tuple
    """
    e = np.array([e1, e2, e3])

    # Normalize: one component must be 0
    min_idx = np.argmin(np.abs(e))
    e = e - e[min_idx]

    # Select correct quadrant
    if min_idx == 0:
        e[2] *= -1
    elif min_idx == 1:
        e[0] *= -1
    else:
        e[1] *= -1

    a, b = (e @ ab_transform[:, :2])
    return full_angle_to_half(a, b)
