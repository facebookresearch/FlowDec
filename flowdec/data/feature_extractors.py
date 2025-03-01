# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import math

import torch
from torch import nn
from torch import Tensor
from typing import Optional, Sequence
from einops import rearrange

from flowdec.util.logging import log


class InvertibleFeatureExtractor(nn.Module, abc.ABC):
    """
    An invertible feature extractor, i.e. a one-to-one mapping that has a forward and a true inverse.
    It should hold up to numerical error that `extractor.invert(extractor(x)) == x`.
    """
    @abc.abstractmethod
    def invert(self, x, **kwargs):
        pass


class AmplitudeCompressedComplexSTFT(InvertibleFeatureExtractor):
    """
    A convenient composition of ComplexSTFT() and CompressAmplitudesAndScale().
    """
    def __init__(
        self,
        window_fn, n_fft, sampling_rate,
        alpha, beta,
        hop_length=None, n_hops=None,
        learnable_window=False,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.complex_stft = ComplexSTFT(
            window_fn, n_fft, sampling_rate, hop_length=hop_length, n_hops=n_hops,
            learnable_window=learnable_window,
        )
        self.compress = CompressAmplitudesAndScale(
            compression_exponent=alpha,
            scale_factor=beta,
        )

    def forward(self, x: Tensor, **kwargs):
        X = self.complex_stft(x, **kwargs)
        out = self.compress(X, **kwargs)
        return out

    def invert(self, X: Tensor, **kwargs):
        X = self.compress.invert(X, **kwargs)
        x = self.complex_stft.invert(X, **kwargs)
        return x


class ComplexSTFT(InvertibleFeatureExtractor):
    def __init__(
            self, window_fn, n_fft, sampling_rate, hop_length=None, n_hops=None, learnable_window=False,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (hop_length is not None) ^ (n_hops is not None),\
            "Exactly one of {hop_length, n_hops} must be specified!"
        if hop_length is None:
            hop_length = int(math.ceil(n_fft / n_hops))
            log.info(f"ComplexSTFT: Converted {n_hops=} into {hop_length=}.")

        window_fn = getattr(torch.signal.windows, window_fn)
        self.learnable_window = learnable_window
        self.window = nn.Parameter(window_fn(n_fft), requires_grad=learnable_window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate

        win_dur = self.n_fft / self.sampling_rate
        hop_dur = self.hop_length / self.sampling_rate
        log.info(f"ComplexSTFT with {win_dur*1000:.2f}ms window, {hop_dur*1000:.2f}ms hop.")

        self.center = True  # not configurable for now - TODO?

    def forward(self, x: Tensor, **kwargs):
        """Assumes x is an audio tensor of shape [B, C, T] or [B, T]"""
        # rearrange() used since stft() API is annoying and only wants *one* extra (batch) dim
        bc = "b c" if x.ndim == 3 else "b"
        X = torch.stft(
            rearrange(x, f"{bc} t -> ({bc}) t"), n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=self.center,
            onesided=True, return_complex=True,
        )
        X = rearrange(X, f"({bc}) f t -> {bc} f t", b=x.shape[0])
        return X

    def invert(self, X: Tensor, orig_length: Optional[int] = None, **kwargs):
        """Assumes X is a (complex) spectrogram tensor of shape [B, C, F, T] or [B, F, T]"""
        # rearrange() used since istft() API is annoying and only wants *one* extra (batch) dim
        bc = "b c" if X.ndim == 4 else "b"
        x = torch.istft(
            rearrange(X, f"{bc} f t -> ({bc}) f t"), n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=self.center,
            onesided=True, return_complex=False,
            length=orig_length,
        )
        x = rearrange(x, f"({bc}) t -> {bc} t", b=X.shape[0])
        return x


class CompressAmplitudesAndScale(InvertibleFeatureExtractor):
    def __init__(self, compression_exponent: float, scale_factor: float, *args, **kwargs):
        super().__init__()
        self.compression_exponent = compression_exponent
        self.scale_factor = scale_factor

    def forward(self, X: Tensor, comp_eps=None, **kwargs):
        """
        Assumes X is a complex STFT (complex spectrogram).
        """
        alpha = self.compression_exponent
        beta = self.scale_factor
        if alpha != 1:
            if comp_eps is not None:
                X = X + comp_eps
            X = X.abs()**alpha * torch.exp(1j * X.angle())
        return X * beta

    def invert(self, X: Tensor, **kwargs):
        """
        Assumes X is an amplitude-compressed and scaled complex STFT.
        """
        alpha = self.compression_exponent
        beta = self.scale_factor
        X = X / beta
        if alpha != 1:
            X = X.abs()**(1/alpha) * torch.exp(1j * X.angle())
        return X


class InvertibleSequential(InvertibleFeatureExtractor):
    def __init__(self, extractors: Sequence[InvertibleFeatureExtractor]):
        super().__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x, **kwargs):
        for e in self.extractors:
            x = e(x, **kwargs)
        return x

    def invert(self, X, **kwargs):
        for e in reversed(self.extractors):
            X = e.invert(X, **kwargs)
        return X


class NoOp(InvertibleFeatureExtractor):
    def forward(self, x, **kwargs):
        return x

    def invert(self, x, **kwargs):
        return x
