# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
from typing_extensions import override
import numpy as np
import scipy as sp
import torch, torchaudio
from torchaudio import transforms as T
from pytorch_lightning.plugins.environments import SLURMEnvironment


# small 'cache' of high-quality resampler instances from various sampling rates to 48kHz
resamplers48000 = defaultdict(
    default_factory=lambda fs: T.Resample(fs, 48000, lowpass_filter_width=256)
)
for fs in (8000, 16000, 32000, 44100, 48000):
    resamplers48000.get(fs)  # populate


def pad_spec(Y, mode="reflection"):
    """
    Pads a spectrogram to a multiple of 64 along the time dimension.
    Used to work around some issues

    Args:
        Y: the spectrogram to be padded
        mode: the padding mode. 'zero', 'reflection' or 'replication'.

    Returns: A 2-tuple of
        - the padded spectrogram Y'
        - a callable that will undo the padding (turn Y' back into Y).
    """
    T = Y.size(-1)
    if T%64 !=0:
        num_pad = 64-T%64
    else:
        num_pad = 0

    if mode == "zero":
        pad2d = torch.nn.ZeroPad2d((0, num_pad, 0,0))
    elif mode == "reflection":
        pad2d = torch.nn.ReflectionPad2d((0, num_pad, 0,0))
    elif mode == "replication":
        pad2d = torch.nn.ReplicationPad2d((0, num_pad, 0,0))
    else:
        raise NotImplementedError("This function hasn't been implemented yet.")
    return pad2d(Y), lambda Y_: Y_[...,:T]


def normalize_noisy(y, mode, x=None):
    """
    Normalizes the inputs y and (optionally) x by absolute value.

    Args:
        y: noisy/corrupted audio
        mode: 'noisy' or 'none'. If 'none', is a no-op. If 'noisy', normalizes both y and x by the same factor, determined by the maximum absolute value of y.

    Returns: A 3-tuple of
        - y', the normalized version of y
        - x', the normalized version of x (may be None if `x=None` was passed at input)
        - The normalization factor `normfac`, fulfilling `y' * normfac = y' and `x' * normfac = x`.
    """
    if mode == "noisy":
        normfac = y.abs().amax(axis=tuple(range(1, y.ndim)), keepdim=True)
    elif mode == "none":
        normfac = 1.0
    else:
        raise ValueError(f"Unknown normalize mode: {mode}!")

    # Avoid NaNs / inftys / huge values when normfac is close to 0.0. In this case it's likely that we
    # are dealing with a silence segment, so we just reset to normfac=1.0 to avoid the problems
    normfac = torch.where(torch.isclose(normfac, torch.zeros_like(normfac)), 1.0, normfac)

    y = y / normfac
    if x is not None:
        x = x / normfac
    return y, x, normfac


def mean_conf_int(data: np.array, confidence=0.95):
    """
    Returns the mean and confidence interval of some provided data (a NumPy array).

    Args:
        data: The data to be analyzed
        confidence: The desired confidence value. 0.95 by default

    Returns: A 2-tuple of
        - The mean
        - The confidence interval
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def n2t(x):
    """NumPy to Torch. Shorthand for `torch.from_numpy(x)`"""
    return torch.from_numpy(x)


def t2n(x):
    """Torch to NumPy. Shorthand for `x.detach().cpu().numpy()`"""
    return x.detach().cpu().numpy()


def flatten(x):
    """Shorthand for `x.reshape(-1)`"""
    return x.reshape(-1)


def batch_broadcast(a, x):
    """Broadcasts a over all dimensions of x, except the batch dimension, which must match."""

    if len(a.shape) != 1:
        a = a.squeeze()
        if len(a.shape) != 1:
            raise ValueError(
                f"Don't know how to batch-broadcast tensor `a` with more than one effective dimension (shape {a.shape})"
            )

    if a.shape[0] != x.shape[0] and a.shape[0] != 1:
        raise ValueError(
            f"Don't know how to batch-broadcast shape {a.shape} over {x.shape} as the batch dimension is not matching")

    out = a.view((x.shape[0], *(1 for _ in range(len(x.shape)-1))))
    return out


def load48000(path):
    """
    Helper function to load an audio file from a given path as a 48kHz signal.
      If the original sampling rate is not 48kHz, will use torchaudio resampling to 48kHz with
      the lowpass_filter_width=256 option.

    This function will return a 3-dimensional tensor of shape (batch=1, channels, duration).
    """
    try:
        au, fs = torchaudio.load(path)
        if au.shape[0] != 1 and au.ndim > 1:
            au = torch.mean(au, dim=0, keepdim=True)
        if fs != 48000:
            resampler = resamplers48000.get(fs)
            if resampler is None:
                resampler = T.Resample(fs, 48000, lowpass_filter_width=256)
            au = resampler(au)

        if au.ndim == 2:
            au = au.unsqueeze(0)
        assert au.ndim == 3

        return au
    except Exception:
        print("FAILED loading", path)
        return None


def get_audiotype_from_basename(basename: str):
    """
    Tries to determine the audiotype of an audio file based on its basename (filename without path).
    Will try to match and return all of ['music', 'sound', 'speech'], in order. Returns the first match.

    Returns None if determining the audiotype failed.
    """
    prefix_offset = 0
    # skip over prefix like "train_set_" and "val_set_"
    if basename.startswith("train_set_") or basename.startswith("val_set_") or basename.startswith("test_set_"):
        prefix_offset = 2
    try:
        prefix = basename.split('_')[prefix_offset]
        if prefix in ('music', 'sound', 'speech'):
            return prefix
    except Exception:
        pass  # ignore, we just couldn't pattern match, so go on to return None
    return None  # could not determine, return None
