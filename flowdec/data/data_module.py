# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from os.path import basename
import warnings
from glob import glob
from typing import Optional, List, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.functional import resample
import pytorch_lightning as pl

from flowdec.util.logging import log


class PairedAudioFiles(Dataset):
    """
    A dataset of audio file pairs (x, y), loadable from various formats and returned as `torch.Tensor`s.
    Typically used as `x` being the clean (target) audio and `y` being the corrupted/noisy/coded audio to be enhanced.
    """
    def __init__(self,
        path: Optional[str],
        sampling_rate: int,  # in Hz = samples
        target_duration: int,  # in seconds
        random_crop: bool,  # perform random or center cropping?
        noisy_path: str = None,
        mode: str = 'folderglob',  # 'folderglob', 'filelist' or 'paired_filelist'
        pairs_delim: str = ',',
        pretend_len: Optional[int] = None,  # pretend the dataset has this len (size). For debugging
    ):
        """
        Construct a new PairedAudioFiles dataset instance.

        Args:
            path: The path to a directory or filelist that contains the (clean) data. See `mode` and `noisy_path`
                below for more details.
            sampling_rate: The target sampling rate this dataset should produce. Will perform resampling to this
                sampling rate for all files loaded from disk that do not match this sampling rate.
            target_duration: The fixed target duration (in seconds) that this dataset should produce. Will be fulfilled
                approximately depending on `sampling_rate`. Cropping or padding is performed to match each output to
                this target duration.
            random_crop: If `True`, will use random cropping to match `target_duration`.
                If `False`, will use center-cropping instead.
            noisy_path: Path to a directory or filelist that contains the noisy data.
                Ignored if `mode == 'paired_filelist'`.
            mode: One of 'paired_filelist', 'filelist', 'folderglob':
                - 'paired_filelist': Treats `path` as a path to a single .txt file, in which each line contains the
                    clean filepath (x) first and the noisy filepath (y) second, separated by the parameter
                    pairs_delim`. When this mode is active, `noisy_path` is ignored.
                - 'filelist': Treats `path` and `noisy_path` as paths to two list files, matching in number of lines
                    and file order.
                - 'folderglob': Treats `path` and `noisy_path` as paths to folders and globs both for '*.wav' files.
            pairs_delim: Used only when `mode == 'paired_filelist'`.
                The delimiter separating the paths of x and y in each line of the filelist at `path`.
            pretend_len:
                (For debugging only) Overrides the length of the dataset to present as a smaller dataset instead.
            """
        self.mode = mode
        self.sampling_rate = sampling_rate
        self.target_duration = target_duration
        self.random_crop = random_crop
        self.pretend_len = pretend_len
        self.pairs_delim = pairs_delim

        if path is None:
            warnings.warn("Passed path=None, this dataset will be empty!")
            self.clean_files, self.noisy_files = [], []
            return

        if self.mode == 'folderglob':
            self.clean_files = list(sorted(glob(os.path.join(path, '*.wav'))))
            self.noisy_files = list(sorted(glob(os.path.join(noisy_path, '*.wav'))))
            assert all(basename(x) == basename(y) for x, y in zip(self.clean_files, self.noisy_files)),\
                "Did not find the same set of files in the clean and noisy folders!"
        elif self.mode == 'filelist':
            with open(path, 'r') as f:
                self.clean_files = f.read().splitlines()
            with open(noisy_path, 'r') as f:
                self.noisy_files = f.read().splitlines()
        elif self.mode == 'paired_filelist':
            with open(path, 'r') as f:
                pairs = [l.split(self.pairs_delim) for l in f.read().splitlines()]
                self.clean_files = [pair[0] for pair in pairs]
                self.noisy_files = [pair[1] for pair in pairs]
        else:
            raise ValueError(
                f"Unknown mode for PairedAudioFiles: {self.mode} !"
                f"Only 'folderglob', 'filelist' and 'paired_filelist' are implemented.")

        # Sanity checks
        n_clean, n_noisy = len(self.clean_files), len(self.noisy_files)
        if not n_clean == n_noisy:
            raise ValueError(
                f"Found {n_clean} clean files but only {n_noisy} noisy files -- lengths must match!")
        if pretend_len is not None and pretend_len > n_clean:
            raise ValueError(f"Passed {pretend_len=} but only have {n_clean} files to begin with!")

    def get(self, i: int, pad_crop: bool, return_basename: bool = False) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]
    ]:
        """
        Get the `i`-th audio from this dataset. `pad_crop` determines if padding/cropping is applied to match the
        stored target duration (necessary/recommended for training). Can optionally `return_basename`.

        Args:
            i: The index of the audio from this dataset. Should be <= len(dataset)-1, may error otherwise.
            pad_crop: Boolean flag that determines if padding/cropping is performed. This should generally be `True`
                for any training/validation steps that use batches of multiple files. When `pad_crop=True`, the way of cropping depends on the flag `self.random_crop`.
            return_basename: Whether to return the basename (of the noisy file y) as a third entry in the result tuple.
                False by default.

        Returns: A 2- or 3-tuple (depending on `return_basename`) of:
            - Clean audio x
            - Corrupted audio y
            (- the basename of y) -- only if `return_basename=True`.
        """
        basename = os.path.basename(self.noisy_files[i])
        x, fs_x = torchaudio.load(self.clean_files[i])
        y, fs_y = torchaudio.load(self.noisy_files[i])

        # TODO generalize to stereo/multichannel?
        if x.shape[0] > 1:
            x = torch.mean(x, dim=0, keepdim=True)
        if y.shape[0] > 1:
            log.warning(f"Found a coded audio {basename} with > 1 channels...? Something off?")
            y = torch.mean(y, dim=0, keepdim=True)

        # Resample inputs to model's target sampling rate when needed.
        # We use a relatively large lowpass filter width as it has relatively little influence
        # on runtime in practice, but can help avoid aliasing artifacts.
        fs_self = self.sampling_rate
        if fs_x != fs_self:
            x, fs_x = resample(x, fs_x, fs_self, lowpass_filter_width=128), fs_self
        if fs_y != fs_self:
            y, fs_y = resample(y, fs_y, fs_self, lowpass_filter_width=128), fs_self

        if x.shape[-1] < y.shape[-1]:
            y = y[..., :x.shape[-1]]
        elif x.shape[-1] > y.shape[-1]:
            raise ValueError(
                f"Misaligned / broken audio files: y cannot be shorter than x! In: "
                f"x={self.clean_files[i]}, y={self.noisy_files[i]}"
            )

        # NOTE: formulae are applicable for subsequent STFT with center=True
        if pad_crop:
            target_samples = self.target_duration * fs_self
            current_len = x.size(-1)
            pad = max(target_samples - current_len, 0)
            if pad == 0:  # cropping, not padding
                if self.random_crop:
                    # extract random part of the audio file
                    start = int(np.random.uniform(0, current_len-target_samples))
                else:
                    # extract central part of the audio file
                    start = int((current_len-target_samples)/2)
                x = x[..., start:start+target_samples]
                y = y[..., start:start+target_samples]
            else:  # padding, not cropping
                # pad audio on both sides if the length T is smaller than num_frames
                x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant', value=0)
                y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant', value=0)

        if return_basename:
            return x, y, basename
        else:
            return x, y

    def __getitem__(self, i):
        return self.get(i, pad_crop=True, return_basename=True)

    def __len__(self):
        if self.pretend_len is not None:
            return self.pretend_len
        else:
            return len(self.clean_files)


class PairedAudioDataModule(pl.LightningDataModule):
    """
    A LightningDataModule for audio file pairs. Wraps multiple instances of `PairedAudioFiles` datasets.
    """
    def __init__(
        self,
        sampling_rate: int,  # in Hz = samples
        target_duration: int,  # in seconds
        batch_size: int,
        num_workers: int = 10,
        pin_memory: bool = True,
        mode: str = 'folderglob',  # 'folderglob', 'filelist' or 'paired_filelist'
        pairs_delim: str = ',',
        train_x: Optional[str] = None,
        train_y: Optional[str] = None,
        valid_x: Optional[str] = None,
        valid_y: Optional[str] = None,
        test_x: Optional[str] = None,
        test_y: Optional[str] = None,
        dataset_kwargs: Optional[dict] = None,
        **kwargs, ## TODO remove
    ):
        """
        Construct a new PairedAudioDataModule datamodule instance.

        Args:
            sampling_rate: The target sampling rate this dataset should produce. Will perform resampling to this
                sampling rate for all files loaded from disk that do not match this sampling rate.
            target_duration: The fixed target duration (in seconds) that this dataset should produce. Will be fulfilled
                approximately depending on `sampling_rate`. Cropping or padding is performed to match each output to
                this target duration.
            batch_size: The target batch size of each dataset in this datamodule.
            num_workers: The number of workers to use for *each* dataset's dataloader
                (e.g. when 10, will use 10 workers for the train dataloader, 10 for validation, 10 for test).
            pin_memory: Flag to use memory pinning for faster copying to GPU. True by default.
            mode: One of 'paired_filelist', 'filelist', 'folderglob':
                - 'paired_filelist': Treats `train_x`, `valid_x`, `test_x` as paths to a single .txt file each, in
                    which each line contains the clean filepath (x) first and the noisy filepath (y) second, separated by the parameter pairs_delim`. When this mode is active, `train_y`, `valid_y`, `test_y` are ignored.
                - 'filelist': Treats `train_x` and `train_y` (and equally `valid_x`/`valid_y`, `test_x`/`test_y`)
                    as paths to two list files, matching in number of lines and file order.
                - 'folderglob': Treats `train_x` and `train_y` (and equally `valid_x`/`valid_y`, `test_x`/`test_y`)
                    as paths to folders and globs both for '*.wav' files.
            pairs_delim: Used only when `mode == 'paired_filelist'`.
                The delimiter separating the paths of x and y in each line of the filelist at `train_x` / `valid_x` /
                `test_x`.
            train_x/train_y/valid_x/valid_y/test_x/test_y:
                All of the above should be paths to either files or folders, depending on `mode`.
                You may pass `None` if you do not need any of the datasets for your purposes.
            dataset_kwargs: Optional keyword arguments to be passed to the `PairedAudioFiles` dataset constructors.
                Note that these will be passed to *all* three dataset instances (train, valid, and test).
                None by default.
        """
        super().__init__()
        self.mode = mode

        # TODO backwards compat -- remove eventually
        if 'clean_dir_train' in kwargs:
            log.warning('Found deprecated arg "clean_dir_train" to PairedAudioDataModule. '
                        'Using backwards compatible mode and overriding train_x, train_y, ...')
            train_x = kwargs.get('clean_dir_train')
            train_y = kwargs.get('noisy_dir_train')
            valid_x = kwargs.get('clean_dir_valid')
            valid_y = kwargs.get('noisy_dir_valid')
            test_x = kwargs.get('clean_dir_test')
            test_y = kwargs.get('noisy_dir_test')

        if mode != 'paired_filelist':
            # clean/noisy must either both be present, or neither
            assert bool(train_x) == bool(train_y)
            assert bool(valid_x) == bool(valid_y)
            assert bool(test_x) == bool(test_y)
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y = test_y
        self.sampling_rate = sampling_rate
        self.target_duration = target_duration

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pairs_delim = pairs_delim
        self.dataset_kwargs = dataset_kwargs or {}

    def setup(self, stage=None):
        shared_kwargs = dict(
            mode=self.mode,
            sampling_rate=self.sampling_rate,
            target_duration=self.target_duration,
            pairs_delim=self.pairs_delim,
            **self.dataset_kwargs
        )

        if stage == 'fit' or stage is None:
            self.train_set = PairedAudioFiles(
                path=self.train_x,
                noisy_path=self.train_y,
                random_crop=True,
                **shared_kwargs,
            )
            self.valid_set = PairedAudioFiles(
                path=self.valid_x,
                noisy_path=self.valid_y,
                random_crop=False,
                **shared_kwargs,
            )
        if stage == 'test' or stage is None:
            self.test_set = PairedAudioFiles(
                path=self.test_x,
                noisy_path=self.text_y,
                random_crop=False,
                **shared_kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def get_evaluation_samples(self, from_set: str, num_samples: int, seed: Optional[int] = None) -> List[
        Tuple[torch.Tensor, torch.Tensor, str]
    ]:
        """
        Gets uncropped samples of varying lengths from any of the datasets. Can be used for evaluation cases when
        cropping/padding is undesired and batching is not needed.

        Args:
            from_set: From which dataset should samples be returned: 'train', 'valid' or 'test'?
            num_samples: Number of samples to be returned.
            seed: Optional random seed to determine the chosen samples.
                If `None`, will use linear spacing across the chosen dataset instead of random sampling.

        Returns:
            A list of 3-tuples as returned by `PairedAudioFiles.get()`: (x, y, basename).
        """
        if from_set == 'train':
            dset = self.train_set
        elif from_set == 'valid':
            dset = self.valid_set
        elif from_set == 'test':
            dset = self.test_set
        else:
            raise ValueError(f"Unknown set: {from_set}")

        if seed is None:
            idxs = np.linspace(0, len(dset)-1, num_samples).astype(np.int32)
        else:
            idxs = np.random.default_rng(seed).choice(range(len(dset)), num_samples).astype(np.int32)

        if len(set(idxs)) != len(idxs):
            warnings.warn(
                f"Cannot get {num_samples} uniform samples from dataset '{from_set}': There seem to be "
                f"fewer files in that dataset in total! Returning with some duplicate indices..."
            )

        sample_list = []
        for i in idxs:
            x, y, basename = dset.get(i, pad_crop=False, return_basename=True)
            # We .unsqueeze(0) here to introduce a fake batch dimension, generally easier to deal with
            x, y = x.unsqueeze(0), y.unsqueeze(0)
            sample_list.append((x, y, basename))
        return sample_list
