# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributed
import torch, torch.nn as nn
import torchaudio
from einops import rearrange
from torch_pesq import PesqLoss


class TorchPESQSpeechLoss(nn.Module):
    only_applies_to = ['speech']

    def __repr__(self):
        return "TorchPESQSpeechLoss()"

    def __init__(self, sampling_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq_loss = PesqLoss(1.0, sampling_rate)  # will resample from sampling_rate to 16kHz

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor):
        # more than 3 dims are unexpected -- error out
        assert x_hat.ndim <= 3
        assert x.ndim <= 3
        # handle if there are multiple channels (or an unsqueezed channel dim) --> treat as batch dim
        if x_hat.ndim == 3:
            x_hat = rearrange(x_hat, "b c t -> (b c) t")
        if x.ndim == 3:
            x = rearrange(x, "b c t -> (b c) t")

        pesq_loss_vals = self.pesq_loss(x, x_hat)
        pesq_loss_vals = pesq_loss_vals.clamp(max=10)  # clip off crazy high values that occur sometimes (why?!)
        return pesq_loss_vals.mean()


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths = [4096, 2048, 1024, 512],
        loss_fn = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
    ):
        super().__init__()
        self.stft_params = [
            dict(
                n_fft=w,
                hop_length=w // 4,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : Tensor
            Estimate signal
        y : Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for s in self.stft_params:
            stft_kw = dict(
                n_fft=s['n_fft'], hop_length=s['hop_length'],
                window=torch.hann_window(s['n_fft']).to(x.device), return_complex=True)
            X = torch.stft(rearrange(x, "b c t -> (b c) t"), **stft_kw).abs()
            Y = torch.stft(rearrange(y, "b c t -> (b c) t"), **stft_kw).abs()
            loss += self.log_weight * self.loss_fn(
                X.clamp(self.clamp_eps).pow(self.pow).log10(),
                Y.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(X, Y)
        return loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0

    Implementation copied and adapted from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """
    def __init__(
        self,
        sampling_rate = 48000,
        n_mels = [10, 20, 40, 80, 160, 320],
        window_lengths = [128, 256, 512, 1024, 2048, 4096],
        loss_fn = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        mel_fmin = [0, 0, 0, 0, 0, 0, 0],
        mel_fmax = [None, None, None, None, None, None, None],
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.window_lengths = window_lengths
        self.hop_lengths = [w//4 for w in self.window_lengths]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow
        # intentionally no module list so that we don't try to load these weights from a checkpoint or such
        # see also the admittedly a bit odd override of to() below
        self.mel_tfs = [
            torchaudio.transforms.MelSpectrogram(
                self.sampling_rate,
                n_fft=window_length, hop_length=hop_length,
                f_min=fmin, f_max=fmax, n_mels=n_mels,
                window_fn=torch.hann_window,
                norm='slaney',  # consistent with audiotools which uses librosa.filters which has norm='slaney' default
            )
            for n_mels, fmin, fmax, window_length, hop_length in zip(
                self.n_mels, self.mel_fmin, self.mel_fmax, self.window_lengths, self.hop_lengths
            )
        ]

    def to(self, *args, **kwargs):
        for mel_tf in self.mel_tfs:
            mel_tf.to(*args, **kwargs)
        super().to(*args, **kwargs)
        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : Tensor
            Estimate signal
        y : Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        for mel_tf in self.mel_tfs:
            x_mels = mel_tf(x)
            y_mels = mel_tf(y)

            if self.log_weight > 0:
                loss += self.log_weight * self.loss_fn(
                    x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                    y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                )
            if self.mag_weight > 0:
                loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss
