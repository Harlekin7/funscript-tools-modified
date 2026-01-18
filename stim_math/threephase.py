"""
Three-phase signal generator for electrostimulation.

This module provides the core mathematical functions for generating
three-phase electrode signals from alpha/beta position coordinates.

The signal generation follows the formula:
    [L, R, 0]^T = P @ ab_transform @ squeeze @ carrier

Where:
    - P is the electrode-to-channel projection matrix
    - ab_transform converts alpha/beta to electrode potentials
    - squeeze is a projection matrix based on position
    - carrier is the sinusoidal carrier signal
"""

import numpy as np
from typing import Tuple, Union

from .transforms import potential_to_channel_matrix, ab_transform


class ThreePhaseSignalGenerator:
    """
    Three-phase signal generator for stereo audio output.

    Generates L, R channel signals from alpha/beta position coordinates
    using the squeeze projection matrix and carrier signal.
    """

    @staticmethod
    def project_on_ab_coefs(
        alpha: Union[float, np.ndarray],
        beta: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate squeeze matrix coefficients from alpha/beta position.

        The squeeze matrix projects the carrier signal based on the
        current position in the phase diagram.

        Args:
            alpha: Alpha coordinate(s), -1 to 1
            beta: Beta coordinate(s), -1 to 1

        Returns:
            Tuple of (t11, t12, t21, t22) squeeze matrix elements
        """
        alpha = np.asarray(alpha, dtype=np.float32)
        beta = np.asarray(beta, dtype=np.float32)

        # Calculate radius from center
        r = np.sqrt(alpha ** 2 + beta ** 2)

        # Clamp to unit circle - positions outside are normalized
        mask = r > 1
        if np.any(mask):
            alpha = np.where(mask, alpha / r, alpha)
            beta = np.where(mask, beta / r, beta)
            r = np.clip(r, 0, 1)

        # Squeeze matrix coefficients
        # These project the 2D carrier onto the electrode configuration
        t11 = (2 - r + alpha) / 2
        t12 = -beta / 2
        t21 = t12  # Symmetric
        t22 = (2 - r - alpha) / 2

        return t11, t12, t21, t22

    @staticmethod
    def carrier(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate carrier signal from phase angle.

        Args:
            theta: Phase angle(s) in radians

        Returns:
            Tuple of (cos(theta), sin(theta)) as float32 arrays
        """
        return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)

    @staticmethod
    def generate(
        theta: np.ndarray,
        alpha: Union[float, np.ndarray],
        beta: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate L, R channel signals.

        This is the main signal generation function that combines
        the carrier signal with the position-dependent squeeze matrix.

        Args:
            theta: Phase angle array (radians)
            alpha: Alpha position (-1 to 1), scalar or array
            beta: Beta position (-1 to 1), scalar or array

        Returns:
            Tuple of (L, R) channel signal arrays
        """
        # Generate carrier
        carrier_x, carrier_y = ThreePhaseSignalGenerator.carrier(theta)

        # Get squeeze matrix coefficients
        t11, t12, t21, t22 = ThreePhaseSignalGenerator.project_on_ab_coefs(alpha, beta)

        # Apply squeeze matrix to carrier
        a = t11 * carrier_x + t12 * carrier_y
        b = t21 * carrier_x + t22 * carrier_y

        # Transform to L, R channels
        T = (potential_to_channel_matrix @ ab_transform)[:2, :2] / np.sqrt(3)
        L, R = T @ np.array([a, b])

        return L.astype(np.float32), R.astype(np.float32)

    @staticmethod
    def electrode_amplitude(
        alpha: Union[float, np.ndarray],
        beta: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate amplitude at each electrode (N, L, R).

        This is used for visualization to show how much signal
        each electrode is receiving at a given position.

        Args:
            alpha: Alpha coordinate(s)
            beta: Beta coordinate(s)

        Returns:
            Tuple of (N, L, R) electrode amplitudes (0 to 1)
        """
        alpha = np.asarray(alpha, dtype=np.float32)
        beta = np.asarray(beta, dtype=np.float32)

        def add_sine(a: float, b: float, phase: float) -> float:
            """Calculate amplitude of a*sin(x) + b*sin(x+phase)"""
            return np.sqrt(a**2 + b**2 + 2*a*b*np.cos(phase))

        # Get squeeze matrix
        t11, t12, t21, t22 = ThreePhaseSignalGenerator.project_on_ab_coefs(alpha, beta)
        squeeze = np.array([[t11, t12],
                           [t21, t22]])

        # Apply ab_transform to get electrode coefficients
        ab_t = np.array([[1, 0],
                        [-0.5, np.sqrt(3)/2],
                        [-0.5, -np.sqrt(3)/2]]) / np.sqrt(3)
        T = ab_t @ squeeze

        # Calculate amplitudes (combining sin and cos components at 90 degree phase)
        N = add_sine(T[0, 0], T[0, 1], np.pi/2)
        L = add_sine(T[1, 0], T[1, 1], np.pi/2)
        R = add_sine(T[2, 0], T[2, 1], np.pi/2)

        return N, L, R

    @staticmethod
    def channel_amplitude(
        alpha: Union[float, np.ndarray],
        beta: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate amplitude of L and R output channels.

        Args:
            alpha: Alpha coordinate(s)
            beta: Beta coordinate(s)

        Returns:
            Tuple of (L_amp, R_amp, center_amp, phase_diff)
        """
        alpha = np.asarray(alpha, dtype=np.float32)
        beta = np.asarray(beta, dtype=np.float32)

        def add_sine(a: float, b: float, phase: float) -> float:
            """Calculate amplitude of a*sin(x) + b*sin(x+phase)"""
            return np.sqrt(a**2 + b**2 + 2*a*b*np.cos(phase))

        def find_phase(a: float, b: float, phase: float) -> float:
            """Calculate phase of a*sin(x) + b*sin(x+phase)"""
            return np.arctan2(a*np.sin(0) + b*np.sin(phase),
                            a*np.cos(0) + b*np.cos(phase))

        # Transformation matrix
        P = np.array([[1, -1, 0],
                     [1, 0, -1],
                     [0, 1, -1]])
        ab_t = np.array([[1, 0],
                        [-0.5, np.sqrt(3)/2],
                        [-0.5, -np.sqrt(3)/2]]) / np.sqrt(3)

        # Get squeeze matrix
        t11, t12, t21, t22 = ThreePhaseSignalGenerator.project_on_ab_coefs(alpha, beta)
        squeeze = np.array([[t11, t12],
                           [t21, t22]])

        T = P @ ab_t @ squeeze

        L = add_sine(T[0, 0], T[0, 1], np.pi/2)
        R = add_sine(T[1, 0], T[1, 1], np.pi/2)
        center = add_sine(T[2, 0], T[2, 1], np.pi/2)
        phase_L = find_phase(T[0, 0], T[0, 1], np.pi/2)
        phase_R = find_phase(T[1, 0], T[1, 1], np.pi/2)

        return L, R, center, np.abs(phase_L - phase_R)

    @staticmethod
    def generate_preview_waveform(
        alpha: float,
        beta: float,
        frequency_hz: float = 1000.0,
        duration_s: float = 0.01,
        sample_rate: int = 44100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a short preview waveform for visualization.

        Args:
            alpha: Alpha position (-1 to 1)
            beta: Beta position (-1 to 1)
            frequency_hz: Carrier frequency in Hz
            duration_s: Duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (time_array, L_signal, R_signal)
        """
        n_samples = int(duration_s * sample_rate)
        t = np.linspace(0, duration_s, n_samples)
        theta = 2 * np.pi * frequency_hz * t

        # Broadcast alpha/beta to match theta length
        alpha_arr = np.full_like(theta, alpha)
        beta_arr = np.full_like(theta, beta)

        L, R = ThreePhaseSignalGenerator.generate(theta, alpha_arr, beta_arr)

        return t, L, R
