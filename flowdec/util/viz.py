# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Visualization utilities. Unused in the main codebase but may be useful for plotting, especially in Jupyter notebooks.
"""

import matplotlib.pyplot as plt, matplotlib as mpl
from IPython.display import display, Audio
import torch
import numpy as np


def plot_spec(spec, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    imshow_kwargs = dict(norm=mpl.colors.LogNorm(), origin='lower', interpolation='none', aspect='auto')
    imshow_kwargs.update(kwargs)
    img = ax.imshow(spec.abs(), **imshow_kwargs)
    return img, ax


def get_spec(audio, sr, stft_params=None):
    if stft_params is None:
        windur, hopdur = 32e-3, 8e-3
        n_fft = int(sr*windur)
        hop_length = int(sr*hopdur)
        stft_params = dict(n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft), return_complex=True)
    spec = torch.stft(audio, **stft_params)
    return spec


def show_as_spec(audio, sr, stft_params=None, ax=None, **kwargs):
    spec = get_spec(audio, sr, stft_params=stft_params)
    if 'extent' not in kwargs:
        kwargs['extent'] = (0, audio.shape[-1]/sr, 0, sr/2/1000)

    newax = ax is None
    img, ax = plot_spec(spec, ax=ax, **kwargs)
    if newax:
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [kHz]")
    return img, ax


def show_as_audio(audio, sr):
    return display(Audio(audio, rate=sr))


def ccmap_img(cimg, amp_tf=lambda a: a, inv=False, mult=True, cmap='hsv', prange=(-np.pi, np.pi), transparent_nans=False):
    """
    Maps a complex array to rgba values with information about amplitude and phase, by using
    `phase_cmap` ('twilight' by default) as the colormap for the phase information and mapping
    the normalized amplitudes to:

        - mult=False: The opacity (i.e., large amplitudes map to high saturation, when the background is white).
        - mult=True: The brightness, by multiplying all RGB channels with the normalized amplitudes.
    """
    cmp = mpl.colormaps.get_cmap(cmap)
    a = np.abs(cimg)
    p = np.angle(cimg)
    pn = (p - prange[0]) / (prange[1] - prange[0])
    phase = cmp(pn)
    if amp_tf is not None:
        a_raw = amp_tf(a)
        a = a_raw
    a = (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))

    if mult:
        rgba = phase
        fac = (1-a) if inv else a
        fac = fac[..., None]
        rgba[...,:3] *= fac
        rgba[...,3] = 1
    else:
        rgba = phase
        rgba[...,3] = (1-a) if inv else a

    if transparent_nans:
        rgba[...,3] = np.where(np.isnan(a_raw), np.nan, rgba[..., 3])
    return rgba