"""
Stim Math module - Three-phase signal generation for electrostimulation.

This module provides mathematical functions for calculating three-phase
electrode signals from alpha/beta position coordinates. The math is
reimplemented based on standard three-phase electrical principles.
"""

from .threephase import ThreePhaseSignalGenerator
from .transforms import (
    potential_to_channel_matrix,
    potential_to_channel_matrix_inv,
    ab_transform,
    ab_transform_inv,
    ab_to_item_pos,
    item_pos_to_ab
)

__all__ = [
    'ThreePhaseSignalGenerator',
    'potential_to_channel_matrix',
    'potential_to_channel_matrix_inv',
    'ab_transform',
    'ab_transform_inv',
    'ab_to_item_pos',
    'item_pos_to_ab'
]
