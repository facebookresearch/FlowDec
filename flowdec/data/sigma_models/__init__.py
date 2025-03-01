# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Models for non-scalar sigma_y in FlowDec.
"""

import os
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from flowdec.util.logging import log


def from_file(filename: str, factor: float = 1.0, kernel_bandwidth: Optional[float] = None) -> np.array:
    """
    Loads a 1-D frequency-wise sigma_y curve from a .npy file. Optionally applies multiplication by a constant factor, and Gaussian kernel smoothing of the curve after loading.

    Args:
        filename: Path to a file that can be loaded by `np.load`.
        factor: Optional factor to multiply the loaded curve with. 1.0 by default (no scaling).
        kernel_bandwidth: Bandwidth of the Gaussian kernel used for smoothing. None by default (no smoothing).

    Returns:
        The loaded, (optionally) scaled and (optionally) smoothed curve as a NumPy array.

    NOTE: The `kernel_bandwidth` is in units of frequency bands, and thus has a different effective meaning when loading
    from a file that has e.g. 768 bands vs. one that has 512 bands.

    NOTE: Two simple such example curve .npy files are provided in the same directory as this Python file.
    """
    if not os.path.isabs(filename):
        filename = os.path.join(os.path.dirname(__file__), filename)
    curve = np.load(filename)
    if kernel_bandwidth is not None:
        # TODO we may want to implement smoothing *before* taking the root, which is closest to
        # what we did in the original experiments. Here we're loading pre-rooted and smoothing that
        curve = gaussian_filter(curve, sigma=kernel_bandwidth, mode='nearest')
    curve = torch.from_numpy(curve).unsqueeze(-1)  # broadcast along time
    log.info(f"Loaded sigma_y curve with {factor=}, {kernel_bandwidth=} from file {filename}")
    return factor * curve


__all__ = ['from_file']
