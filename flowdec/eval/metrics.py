# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
import torch
from torch import nn
import torch.distributed
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import tqdm
import librosa

from pystoi import stoi as calc_stoi
from pesq import pesq as calc_pesq
from speechmos import dnsmos

with warnings.catch_warnings(record=False):  # mute warnings from pysepm import
    warnings.simplefilter("ignore")
    from pysepm.qualityMeasures import SNRseg, fwSNRseg

from flowdec.util.other import flatten, t2n, load48000
from flowdec.util.logging import log
from flowdec.eval.sigmos.sigmos import SigMOS
from flowdec.eval.visqol import ViSQOL as ViSQOLWrapper


def safe_flatten(x_orig):
    if x_orig is None:
        return x_orig
    x = x_orig.squeeze()
    if x.ndim != 1:
        warnings.warn(
            f"Found not-1dim-squeezable tensor with shape: {x_orig.shape}. "
            f"Flattening anyways.")
    return flatten(x)


def get_metrics_row(metrics, row_name, x_hat, x, y, meta=None):
    row = {**(meta or {}), **{'name': row_name}}

    for metric in metrics:
        if isinstance(metric, tuple):
            metric, namefilter = metric
        else:
            namefilter = None

        try:
            if namefilter is None or namefilter in row_name:
                #start = time.time()
                result = metric(x_hat, x, y, row_name)
                #end = time.time()
                #print(f"{(end - start):.3f} taken for {str(metric)[:20]}")

                if len(metric.names) == 1:
                    result = [result]
                for name, value in zip(metric.names, result):
                    row[name] = value
            else:
                for name in metric.names:
                    row[name] = np.nan
        except Exception as e:
            log.exception(f"Exception occurred calculating metric {metric}. Returning NaNs for this")
            for name in metric.names:
                row[name] = np.nan
    return row


def get_metrics_df(x_hats, xs, ys, metrics, names=None,
                   crop_to_x=False, crop_to_x_hat=False, progress=True, meta=None):
    results = []
    assert len(x_hats) == len(xs)
    assert len(ys) == len(xs)

    eval_list = list(enumerate(zip(x_hats, xs, ys)))
    if progress:
        eval_list = tqdm.tqdm(eval_list, unit="file")
    for i, (x_hat, x, y) in eval_list:
        try:
            x_hat, x, y = _load_if_path(x_hat), _load_if_path(x), _load_if_path(y)
            if crop_to_x:
                x_hat = x_hat[..., :x.shape[-1]]
                y = y[..., :x.shape[-1]]
            if crop_to_x_hat:
                x = x[..., :x_hat.shape[-1]]
                y = y[..., :x_hat.shape[-1]]

            name = names[i] if names is not None else str(i)
            meta_ = meta[i] if meta is not None else None
            results.append(get_metrics_row(metrics, name, x_hat, x, y, meta=meta_))
        except Exception:
            print("Exception when processing", names, "-- skipping")

    if not len(results):
        raise ValueError("Produced an empty DataFrame!")
    return pd.DataFrame(results)


# This is written as a class so we can easily pickle it for use with Python multiprocessing
# The more natural lambdas/functools.partial are unfortunately much more difficult to deal with w/ multiprocessing
class InitializeMetrics:
    def __init__(self, sr, visqol, mask=None):
        self.sr = sr
        self.visqol = visqol
        self.mask = mask

    def __call__(self, id_=None):
        id_ = id_ if id_ is not None else 0
        cuda_device = id_ % torch.cuda.device_count()
        torch_device = f'cuda:{cuda_device}'
        print(f"Initialized metrics for ID {id_}, using CUDA device {cuda_device}")

        sr = self.sr
        visqol = self.visqol
        metrics = [
            SISXR(sr),
            LogSpecMSE(sr).to(torch_device),
            (PESQ(sr), 'speech'),
            (SIGMOS(sr, cuda_device=cuda_device), 'speech'),
            FrequencyWeightedSegmentalSNR(sr),
            SegmentalSNR(sr)
        ]
        if visqol:
            visqol_au = ViSQOL(sr, backend='bindings', force_mode='audio')
            visqol_au.names = ['visqol_mos_audio']
            visqol_sp = ViSQOL(sr, backend='bindings', force_mode='speech')
            visqol_sp.names = ['visqol_mos_speech']
            metrics += [visqol_au, (visqol_sp, 'speech')]

        if self.mask is not None:
            return [m for (m, mask) in zip(metrics, self.mask) if mask]
        else:
            return metrics


# --- A bit of absurd code for Python multiprocessing below...

def _process_triple(metrics, name, x_hat, x, y, crop_to_x, crop_to_x_hat, meta):
    """
    The worker function for parallel metrics evaluation.
    Processes a triple (x_hat, x, y) with all metrics and options and returns an evaluated row with results.
    """
    try:
        x_hat, x, y = _load_if_path(x_hat), _load_if_path(x), _load_if_path(y)
        if crop_to_x:
            x_hat = x_hat[..., :x.shape[-1]]
            y = y[..., :x.shape[-1]]
        if crop_to_x_hat:
            x = x[..., :x_hat.shape[-1]]
            y = y[..., :x_hat.shape[-1]]
        return get_metrics_row(metrics, name, x_hat, x, y, meta=meta)
    except Exception as e:
        print(f"Exception when processing {name} -- skipping: {str(e)}")
        return None

def process_triple_star(args):
    """
    Small wrapper around _process_triple so we can use pool.imap() instead of starmap(),
    which is better for use with tqdm progress bars.

    Passes the global _metrics variable to _process_triple as the first arg.
    """
    return _process_triple(*((_metrics,) + args))

# global variable storing currently used metrics. Only used with parallelized evaluation.
_metrics = None
# global variable storing the unique increasing ID of the worker process. Only used with parallel evaluation.
_worker_id = None

class Initializer:
    """
    Tiny pickleable helper for parallel Pool worker initialization.
    """
    def __init__(self, init_metrics_fn):
        """
        Args:
        init_metrics_fn: A callable with the argument "id_" that returns a list of initialized metrics.
            Should make use of "id_" as necessary (e.g. to determine the CUDA device a metric instance should be put)
        """
        self.init_metrics_fn = init_metrics_fn

    def __call__(self, worker_id_queue):
        global _metrics, _worker_id
        _worker_id = worker_id_queue.get()
        _metrics = self.init_metrics_fn(id_=_worker_id)

def get_metrics_df_parallel(
        x_hats, xs, ys, init_metrics_fn: InitializeMetrics, poolsize: int = 96,
        names=None, crop_to_x: bool = False, crop_to_x_hat: bool = False, progress=True, meta=None):
    """
    Parallel variant of `get_metrics_df` that uses a multiprocessing.Pool to process the metrics in parallel,
    parallelized over all triples (x_hat, x, y).

    Note that the args (x_hats, xs, ys, names, meta) MUST have the same ordering, since they will be zipped together.

    Args:
        x_hats: The list of estimate audios (Tensors) or file paths to these (strings).
        xs: The list of ground-truth audios (Tensors) or file paths to these (strings).
        ys: The list of initial-estimate audios from the first-stage codec (Tensors) or file paths to these (strings).
            Pass the same list as `x_hats` if this does not match the evaluation scenario (e.g. evaluating a baseline).
        init_metrics_fn: A callable that returns initialized metrics when called with an "id_" parameter, see `InitializeMetrics`
        poolsize: The size of the Pool / number of worker processes
        names: List of file names. May be used for determining if a metric applies to a given audio or not.
        crop_to_x: If True, will crop x_hat and y to the length of x.
            Useful if some estimates are slightly longer than x.
        crop_to_x_hat: If True, will crop x and y to the length of x_hat (after crop_to_x is executed).
            Useful if some estimates are slightly shorter than x.
        progress: Pass True to show a tqdm progress bar.
        meta: (Optional) a list of dicts matching x_hats/xs/ys containing meta information such as file paths.
            This will be included in each respective row of the output dataframe.
    """
    assert len(x_hats) == len(xs)
    assert len(ys) == len(xs)
    eval_list = list(enumerate(zip(x_hats, xs, ys)))
    names = [names[i] if names is not None else str(i) for i in range(len(eval_list))]
    metas = [meta[i] if meta is not None else None for i in range(len(eval_list))]

    n = poolsize
    import torch, torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    q = torch.multiprocessing.Queue(n)
    for i in range(n):
        q.put(i)
    with torch.multiprocessing.Pool(n, initializer=Initializer(init_metrics_fn), initargs=(q,)) as pool:
        results = []
        argslist = [
            (name, x_hat, x, y, crop_to_x, crop_to_x_hat, meta)
            for (i, (x_hat, x, y)), name, meta in zip(eval_list, names, metas)
        ]
        iterator = pool.imap(process_triple_star, argslist)
        if progress:
            iterator = tqdm.tqdm(iterator, unit="file", total=len(argslist))
        for result in iterator:
            results.append(result)
    # Remove any Nones that were added due to exceptions
    results = [result for result in results if result is not None]

    if not len(results):
        raise ValueError("Produced an empty DataFrame!")
    return pd.DataFrame(results)


# --- Absurd code over; here's some definitions of various metrics


class Metric(nn.Module):
    def __init__(self, sr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr = sr


class SISXR(Metric):
    names = ["sisdr", "sisir", "sisar"]

    def forward(self, x_hat, x, y, name=None):
        x_hat, x, y = [t2n(safe_flatten(t)) for t in (x_hat, x, y)]
        n = y - x
        # Try to correct for at least a global phase flip between x and y:
        #   n is likely the variant with smaller power
        if _norm2(y + x) < _norm2(y - x):
            n = y + x
        s_target, e_noise, e_art = si_sxr_components(x_hat, x, n)
        si_sdr = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise + e_art)**2)
        si_sir = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise)**2)
        si_sar = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_art)**2)
        return si_sdr, si_sir, si_sar


class ESTOI(Metric):
    names = ["estoi"]

    def forward(self, x_hat, x, y=None, name=None):
        # (E)STOI does its own resampling
        return calc_stoi(
            t2n(safe_flatten(x)),
            t2n(safe_flatten(x_hat)),
            self.sr,
            extended=True
        )


class PESQ(Metric):
    names = ["pesq"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resampler = T.Resample(self.sr, 16000)

    def forward(self, x_hat, x, y=None, name=None):
        # We have to resample to 16000 Hz for PESQ (no-op if self.sampling_rate==16000)
        return calc_pesq(
            16000,
            t2n(self.resampler(safe_flatten(x))),
            t2n(self.resampler(safe_flatten(x_hat))),
            'wb'
        )


class DNSMOS(Metric):
    names = ["ovrl_mos", "sig_mos", "bak_mos", "p808_mos"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resampler = T.Resample(self.sr, 16000)

    def forward(self, x_hat, x=None, y=None, name=None):
        result_dict = dnsmos.run(t2n(self.resampler(safe_flatten(x_hat))), sr=16000)
        return [result_dict[name] for name in self.names]


class SIGMOS(Metric):
    names = [
        'MOS_OVRL', 'MOS_SIG', 'MOS_NOISE',
        'MOS_COL', 'MOS_DISC', 'MOS_LOUD', 'MOS_REVERB',
    ]

    def __init__(self, sr, cuda_device=0, *args, **kwargs):
        super().__init__(sr, *args, **kwargs)
        self.sigmos_instance = SigMOS(
            model_dir=os.path.join(os.path.dirname(__file__), "sigmos"),
            cuda_device=cuda_device
        )

    def forward(self, x_hat, x=None, y=None, name=None):
        result_dict = self.sigmos_instance.run(t2n(safe_flatten(x_hat)), sr=self.sr)
        return [result_dict[name] for name in self.names]


class LogSpecMSE(Metric):
    names = ["logspec_mse"]

    def __init__(self, sr, *args, win_dur=32e-3, hop_dur=8e-3, win_fn='hann', eps=1e-8, **kwargs):
        super().__init__(sr, *args, **kwargs)
        self.win_dur = win_dur
        self.hop_dur = hop_dur
        self.n_fft = int(self.win_dur * self.sr)
        self.hop_length = int(self.hop_dur * self.sr)
        self.win_fn = win_fn
        self.window_fn = getattr(torch.signal.windows, win_fn)
        self.eps = eps
        self.spec_transform = T.Spectrogram(
            n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length,
            center=True, window_fn=self.window_fn, power=2)
        self.device = None

    def to(self, device, *args, **kwargs):
        self.device = device
        super().to(device, *args, **kwargs)
        self.spec_transform = self.spec_transform.to(device, *args, **kwargs)
        return self

    def _prep_inputs(self, x_hat, x):
        with torch.inference_mode():
            x_hat = safe_flatten(x_hat)
            x = safe_flatten(x)
            if self.device is not None:
                x_hat = x_hat.to(self.device)
                x = x.to(self.device)
            return x_hat, x

    def forward(self, x_hat, x, y=None, name=None):
        with torch.inference_mode():
            x_hat, x = self._prep_inputs(x_hat, x)
            spec_x_hat = self.spec_transform(x_hat).abs()
            spec_x = self.spec_transform(x).abs()
            logspec_x_hat = 10*torch.log10(torch.clamp(spec_x_hat, min=self.eps))
            logspec_x = 10*torch.log10(torch.clamp(spec_x, min=self.eps))
            return torch.mean(torch.square(torch.abs(logspec_x - logspec_x_hat))).item()


def visqol_mode_heuristic(name):
    if name is None:
        return None

    if 'speech' in name[:30]:
        return 'speech'
    elif 'sound' in name[:30]:
        return 'audio'
    elif 'music' in name[:30]:
        return 'audio'
    else:
        return None


def get_visqol_api(which):
    # ViSQOL uses custom C bindings which require various odd imports...
    # We import these inside this function so the rest of the code still works when we do not have ViSQOL installed
    # as a Python package.
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    from visqol.pb2 import similarity_result_pb2  # needed import even if the symbol itself is unused!!
    similarity_result_pb2;  # just so the above line is not removed by some code cleaners etc.

    # Adapted from README at https://github.com/google/visqol
    if which == 'audio':
        config = visqol_config_pb2.VisqolConfig()
        config.audio.sample_rate = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
        config.options.svr_model_path = os.path.join(
            os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
    elif which == 'speech':
        config = visqol_config_pb2.VisqolConfig()
        config.audio.sample_rate = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
        config.options.svr_model_path = os.path.join(
            os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
    else:
        raise ValueError()

    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    return api


class ViSQOL(Metric):
    names = ["visqol_mos"]

    def __init__(self, sr, visqol_folder=None, backend='subprocess', force_mode=None, *args, **kwargs):
        """
        Initialize the ViSQOL metric.

        Args:

        visqol_folder: Path to an installed and built ViSQOL folder. Only needed when passing `backend=subprocess`.

        backend: The kind of backend that implements ViSQOL to use. Three are supported:
          - subprocess: Calls out to an external `visqol` binary. This requires `visqol_folder` to be passed to this constructor as well. May be a bit slow since it has to deal with subprocesses.
          - bindings: Uses the Python C bindings from the official ViSQOL library [1]. Requires these bindings to be installed. Should be fastest. Does not work with multiprocessing since these bindings cannot be pickled.
          - bindings_on_the_fly: Also uses the Python C bindings [1], but constructs the API objects on every call. This can be used with multiprocessing since the bindings do not need to be pickled. May be slower than `bindings` due to this reinitialization on every call.

        force_mode: 'audio' or 'speech'. Forces the ViSQOL mode to be this value on every call.
          If not passed, will heuristically auto-determine the type of audio based on the provided filename and choose the ViSQOL mode based on this. Passing force_mode and using two instances (one for speech, one for audio) is generally more reliable.

        [1]: https://github.com/google/visqol/tree/b2b2a64dd8bf1378f9aa1119cc7b36a8bcda3757/python
        """

        super().__init__(sr, *args, **kwargs)
        assert backend in ('subprocess', 'bindings', 'bindings_on_the_fly')
        self.backend = backend
        if self.backend == 'subprocess':
            assert visqol_folder is not None
        assert force_mode in (None, 'audio', 'speech')
        self.force_mode = force_mode

        self.init_success = False
        if backend == 'subprocess':
            try:
                self.visqol_obj_audio = ViSQOLWrapper(
                    visqol_folder, mode='audio',
                    model="tcdaudio14_aacvopus_coresv_svrnsim_n.68_g.01_c1.model",
                    use_lattice_model=False)
                self.visqol_obj_speech = ViSQOLWrapper(
                    visqol_folder, mode='speech',
                    model="lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite",
                    use_lattice_model=True)
                self.init_success = True
            except Exception as e:
                print(f"Failed to initialize ViSQOL, will return NaNs instead: {e}")
        elif backend == 'bindings':
            self.apiA = get_visqol_api('audio')
            self.apiS = get_visqol_api('speech')
            self.init_success = True
        elif backend == 'bindings_on_the_fly':
            self.init_success = True

    def forward(self, x_hat, x, y=None, name=None, mode_override=None):
        if not self.init_success:
            return np.nan

        if self.force_mode is not None:
            visqol_mode = mode_override if mode_override is not None else self.force_mode
        else:
            visqol_mode = mode_override if mode_override is not None else visqol_mode_heuristic(name)
            if visqol_mode is None:
                visqol_mode = 'audio'
                log.warning(f"Heuristically determining ViSQOL mode failed for file with name: {name}. "
                            f"Defaulting to audio mode (48kHz).")

        if self.backend == 'subprocess':
            visqol_obj = self.visqol_obj_speech if visqol_mode == 'speech' else self.visqol_obj_audio
            # We pass pad_with_silence=True to follow the official recommendations.
            # There is no .item() call, since visqol_obj returns a float
            return visqol_obj(x, x_hat, sr=self.sr, pad_with_silence=True)
        elif self.backend.startswith('bindings'):
            x, x_hat = flatten(t2n(x)).astype(np.float64), flatten(t2n(x_hat)).astype(np.float64)

            if visqol_mode == 'speech':
                apiS = self.apiS if self.backend == 'bindings' else get_visqol_api('speech')

                with warnings.catch_warnings(record=False):  # mute warnings from old ViSQOL API
                    warnings.simplefilter("ignore")
                    result = apiS.Measure(
                        librosa.resample(x, orig_sr=self.sr, target_sr=16000, axis=-1),
                        librosa.resample(x_hat, orig_sr=self.sr, target_sr=16000, axis=-1)
                    ).moslqo
                    return result
            else:
                apiA = self.apiA if self.backend == 'bindings' else get_visqol_api('audio')
                with warnings.catch_warnings(record=False):  # mute warnings from old ViSQOL API
                    warnings.simplefilter("ignore")
                    result = apiA.Measure(x, x_hat).moslqo
                    return result


class FrequencyWeightedSegmentalSNR(Metric):
    """
    Thin wrapper around `pysepm.qualityMetrics.fwSNRseg`.
    """
    names = ["fwSSNR"]

    def __init__(self, sr, *args, **kwargs):
        super().__init__(sr, *args, **kwargs)

    def forward(self, x_hat, x, y=None, name=None):
        x = t2n(x.reshape(-1, x.shape[-1]))
        x_hat = t2n(x_hat.reshape(-1, x_hat.shape[-1]))
        per_channel = [
            fwSNRseg(x[i], x_hat[i], fs=self.sr, frameLen=0.03, overlap=0.75)
            for i in range(x.shape[0])
        ]
        return np.mean(per_channel)


class SegmentalSNR(Metric):
    """
    Thin wrapper around `pysepm.qualityMetrics.SNRseg`.
    """

    names = ["SSNR"]

    def __init__(self, sr, *args, **kwargs):
        super().__init__(sr, *args, **kwargs)

    def forward(self, x_hat, x, y=None, name=None):
        x = t2n(x.reshape(-1, x.shape[-1]))
        x_hat = t2n(x_hat.reshape(-1, x_hat.shape[-1]))
        per_channel = [
            SNRseg(x[i], x_hat[i], fs=self.sr, frameLen=0.03, overlap=0.75)
            for i in range(x.shape[0])
        ]
        return np.mean(per_channel)


def _norm2(x):
    return np.linalg.norm(x.reshape(-1), ord=2)


def si_sxr_components(s_hat, s, n):
    # s_target
    alpha_s = np.dot(s_hat, s) / np.linalg.norm(s)**2
    s_target = alpha_s * s
    # e_noise
    alpha_n = np.dot(s_hat, n) / np.linalg.norm(n)**2
    e_noise = alpha_n * n
    # e_art
    e_art = s_hat - s_target - e_noise
    return s_target, e_noise, e_art


def _load_if_path(tensor_or_path):
    if isinstance(tensor_or_path, torch.Tensor):
        return tensor_or_path
    else:
        # Load audio file as 48kHz from path
        return load48000(tensor_or_path)
