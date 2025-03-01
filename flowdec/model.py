# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Callable, List, Optional, Tuple, Dict, Union, List, Any
import warnings

import numpy as np
from einops import repeat, parse_shape
import omegaconf
import wandb
import torch
import torch.distributed
import torch.nn as nn
import pytorch_lightning as pl
from torchcfm import ConditionalFlowMatcher
from torchdyn.core import NeuralODE
import pandas as pd

from flowdec import sampling
from flowdec.data.feature_extractors import InvertibleFeatureExtractor
from flowdec.util.logging import log
from flowdec.util.hydra import instantiate_core_objects
from flowdec.util.other import pad_spec, normalize_noisy
from flowdec.sampling.solvers import get_solver

from flowdec.util.other import pad_spec, normalize_noisy
from flowdec.eval.metrics import Metric, get_metrics_row


METRIC_IGNORE_KEYS = set(['name'])


class EnhancementModel(abc.ABC, pl.LightningModule):
    """
    Parent class for the enhancement models:

    * `FlowModel`: a flow-based generative enhancement model such as FlowDec.
    * `ScoreModel`: a score-based generative enhancement model such as SGMSE+.
    * `RegressionModel`: a simple predictive regression enhancement model, e.g. trained with L2.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_extractor: InvertibleFeatureExtractor,
        sampling_rate: int,
        lr: float,
        normalize_mode: str = 'noisy',  # 'noisy' or 'none'
        optimizer_init: Optional[Callable[[List[nn.Parameter], float], torch.optim.Optimizer]] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        eval_metrics: Optional[List[Metric]] = None,
        eval_variants: Optional[Dict[str, Dict[str, Any]]] = None,
        num_eval_files: int = 20,
        full_config: Optional[dict] = None,
        evaluation_seed: Optional[int] = None,
    ):
        """
        Create a new EnhancementModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model ('ncsnpp' by default).
            feature_extractor: An InvertibleFeatureExtractor instance. Use `NoOp` if none is desired.
            sampling_rate: The sampling rate of the audio files this model is trained for.
            lr: The learning rate for the optimizer.
            normalize_mode: Normalize the input waveforms?
                * 'noisy': By the maximum amplitude of the noisy/corrupted estimate y
                * 'none': Do not normalize the input
            optimizer_init: A partial function expecting a list of parameters and a learning rate,
                and returning a torch.optim.Optimizer instance.
            datamodule: A LightningDataModule used for training, we need access to this for the validation
                step as we want to run the model on full files from the plain dataset and not just crops.
            eval_metrics: A list of evaluation metrics to run. Will be used during validation.
            eval_variants: An optional dict mapping variant names (keys) to kwarg dicts (values),
                which are to be passed to `self.enhance` during evaluation. Useful for evaluating different
                kinds of enhancement settings (e.g. samplers, NFEs) during validation. The names will be used
                as suffixes when logging metrics, e.g., "pesq_N=1" when using "N=1" as a key.
            num_eval_files: The number of files to use for evaluation (validation_step) during training
                (20 by default).
            full_config: The full configuration as a dictionary (or other serializable structure), which
                should be written to the checkpoint. Passing *everything* (not just the model config)
                makes it easiest / most reproducible to load from checkpoint later.
            evaluation_seed: The random seed used for determining the set of evaluation files, used during validation.
                If None, will do a uniformly spaced deterministic choice of indices from the validation set.
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.normalize_mode = normalize_mode
        assert self.normalize_mode in ("noisy", "none")
        self.lr = lr

        self.backbone = backbone
        self.feature_extractor = feature_extractor
        self.optimizer_init = optimizer_init if optimizer_init is not None else torch.optim.Adam
        self.datamodule = datamodule
        # unmarshal, we assume we have received a marshalled config (to avoid instantiation)
        self.full_config = omegaconf.OmegaConf.create(full_config)
        if self.full_config is None:
            warnings.warn("Did not pass full_config! Saving to checkpoint will lose config information!")
        if self.datamodule is None:
            warnings.warn("Did not pass datamodule! Validation step will not work.")
        if self.optimizer_init is None:
            warnings.warn("Did not pass optimizer_init! Will default to Adam.")

        self.num_eval_files = num_eval_files
        self.eval_metrics = eval_metrics
        if eval_variants is None:
            eval_variants = [{"name": None, "enhance_kwargs": {}, "every_n_epochs": 1}]
        for eval_variant in eval_variants:
            assert not ("every_n_epochs" in eval_variant and "every_n_steps" in eval_variant),\
                "For eval_variants, pass either every_n_epochs or every_n_steps, but not both!"
        self.eval_variants = eval_variants
        self.evaluation_seed = evaluation_seed

        # Do not pass to logger, we take care of logging the config ourselves, since it's logger-dependent
        self.save_hyperparameters(self.full_config, logger=False)

    # === Optimizer config ===

    def configure_optimizers(self):
        optimizer = self.optimizer_init(self.parameters(), lr=self.lr)
        return optimizer

    # === Data representation helpers ===

    def _preprocess(self, y, x=None, comp_eps=None):
        """
        Preprocesses time-domain waveforms `y` and `x` into a feature representation by

          1. normalizing the amplitudes depending on `self.normalize_mode`
          2. feeding these waveforms to the `feature_extractor` passed at instantiation.

        Returns a 3-tuple of
            - Y: feature representation of y
            - X: feature representation of x
            - preprocess_info: A dictionary containing information about the performed preprocessing, to be used with `self._postprocess` in order to later invert back to waveforms.
        """
        assert x is None or x.shape == y.shape
        # Make inputs have at least 3 dims by unsqueezing repeatedly if needed.
        # We will squeeze those dims back down in _postprocess.
        squeeze_dims = 0
        while y.ndim < 3:
            y = y.unsqueeze(0)
            x = x.unsqueeze(0) if x is not None else x
            squeeze_dims += 1

        y, x, normfac = normalize_noisy(y, mode=self.normalize_mode, x=x)
        Y = self.feature_extractor(y, comp_eps=comp_eps)
        Y, undo_pad_fn = pad_spec(Y, mode="zero")

        X = None
        if x is not None:
            X = self.feature_extractor(x, comp_eps=comp_eps)
            X, _ = pad_spec(X, mode="zero")
            assert X.shape == Y.shape

        orig_length = y.shape[-1]
        preprocess_info = dict(
            orig_length=orig_length, normfac=normfac, undo_pad_fn=undo_pad_fn, squeeze_dims=squeeze_dims)
        return Y, X, preprocess_info

    def _postprocess(self, X, preprocess_info, inv_kwargs=None, batch_filter=None):
        """
        Inverts the preprocessing performed by `self._preprocess` for any given feature representation `X`.

        Args:
            - X: The feature representation to invert back to a waveform.
            - preprocess_info: The preprocessing info dict as returned by `self._preprocess`.

        Extra args:
            - inv_kwargs: Optional dict of extra keyword-args to be passed to `self.feature_extractor.invert`. Usually not needed, only for trying special things out.
            - batch_filter: Optional list to mask out some of the values from the batch dimension of `X`. For example, pass `[True, False, False, True]` to turn a (4, ...)-shaped tensor into a (2, ...)-shaped tensor containing only the first and last elements of the batch. Usually not needed, only used by per-audiotype auxiliary single- and multistep finetuning losses.
        """
        I = preprocess_info
        assert {'orig_length', 'normfac', 'undo_pad_fn', 'squeeze_dims'} <= I.keys()
        undo_pad_fn, orig_length, normfac = I['undo_pad_fn'], I['orig_length'], I['normfac']
        squeeze_dims = I['squeeze_dims']

        X = undo_pad_fn(X)
        x = self.feature_extractor.invert(X, orig_length=orig_length, **(inv_kwargs or {}))
        if batch_filter is not None:
            x = x[batch_filter]
            normfac = normfac[batch_filter]
        x = x * normfac
        for _ in range(squeeze_dims):
            x = x.squeeze(0)
        return x

    # === Abstract methods each subclass must implement ===

    @abc.abstractmethod
    def _loss(self, batch: torch.Tensor, batch_idx: int, which: str) -> float:
        """
        Performs the bulk of a training/validation step and returns a single loss value
            (as a 0-dim Tensor, so functionally a single float value).
        Will be called inside default impl of training_step() and validation_step().
        """
        pass

    @abc.abstractmethod
    def enhance(self, corrupt_batch: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Enhance a (batch of) waveform(s). Output has the same shape as the input.
        """
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    # === Model training and evaluation ===

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch, batch_idx, 'train')
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._loss(batch, batch_idx, 'valid')
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch[0].shape[0])

        # Evaluate audio enhancement performance and log audios to W&B
        if batch_idx == 0 and self.num_eval_files != 0:
            # We ignore the batch (which is usually time-cropped) and get a fixed-size set of uncropped audios instead
            # from a special method of the datamodule
            eval_audios = self.datamodule.get_evaluation_samples(
                'valid', self.num_eval_files, seed=self.evaluation_seed)
            # Evaluate each variant and log each metric with an appropriate suffix
            for variant_dict in self.eval_variants:
                variant_name = variant_dict['name']
                variant_kwargs = variant_dict['enhance_kwargs']
                every_n_epochs = variant_dict.get('every_n_epochs', 1)
                every_n_steps = variant_dict.get('every_n_epochs', 0)
                if every_n_epochs > 0 and self.current_epoch % every_n_epochs != 0:
                    log.debug(f"Skipping eval variant {variant_name}, only every {every_n_epochs} epochs...")
                    continue
                if every_n_steps > 0 and self.global_step % every_n_steps != 0:
                    log.debug(f"Skipping eval variant {variant_name}, only every {every_n_steps} steps...")
                    continue

                log.info(f"Running eval variant {variant_name}...")
                suffix = '' if not variant_name else f'_{variant_name}'
                metric_df, audios = self.evaluate_model(audios=eval_audios, **variant_kwargs)
                for metric_key in set(metric_df.columns) - METRIC_IGNORE_KEYS:
                    metric_values = np.array(metric_df[metric_key])
                    if np.issubdtype(metric_values.dtype, np.number):
                        log.info(f'{metric_key}{suffix} = {np.nanmean(metric_values)}')
                        if np.any(np.isnan(metric_values)):
                            log.warn(
                                f"NaN encountered during eval in metric {metric_key} "
                                f"for {sum(np.isnan(metric_values))}of {len(metric_values)} values!")
                        self.log(f'{metric_key}{suffix}', np.nanmean(metric_values),
                                on_step=False, on_epoch=True, sync_dist=True,
                                batch_size=batch[0].shape[0])
                    else:
                        log.debug(f"Not logging 'metric' {metric_key}: Apparently not a number type")
                # Also log audios as a table (if using WandbLogger)
                if isinstance(self.logger, pl.loggers.WandbLogger):
                    table_columns, table_data = self._get_wandb_audio_table(audios)
                    try:
                        self.logger.log_table(key=f"audios{suffix}", columns=table_columns, data=table_data)
                    except wandb.errors.CommError as e:
                        # we catch and log this exception here in order to avoid crashing the whole training ru
                        # when it happens
                        log.exception("Comm error with W&B trying to log audios!")
                else:
                    log.info("Not logging audios - only available for WandbLogger")

        return loss

    def _get_wandb_audio(self, audio: torch.Tensor) -> wandb.Audio:
        """
        Small helper to fix tensor shapes of the input `audio` tensor, and turn it into a `wandb.Audio` instance
        that can be logged to W&B.
        """
        if audio.ndim > 3:
            raise ValueError("Too large dimensionality of audio!")
        elif audio.ndim == 3:
            audio = audio[0,0]
        elif audio.ndim == 2:
            audio = audio[0]

        return wandb.Audio(audio, sample_rate=self.sampling_rate)

    def _get_wandb_audio_table(self, audios: List[Dict[str, Union[str, torch.Tensor]]]):
        """
        Constructs a W&B audio table from a list of dictionaries, each dict containing the keys:
        - name: Name of this audio row (a string)
        - x_hat: torch.Tensor containing the audio estimate waveform
        - x: torch.Tensor containing the clean audio waveform
        - y: torch.Tensor containing the coded/noisy audio waveform
        """
        table_columns = ["name", "x_hat", "x", "y"]
        table_data = [
            [
                audios[i]["name"],
                *[self._get_wandb_audio(audios[i][key]) for key in table_columns[1:]]
            ]
            for i in range(len(audios))
        ]
        return table_columns, table_data

    def evaluate_model(
            self,
            audios: List[Tuple[torch.Tensor, torch.Tensor, str]],
            metrics: Optional[List[Metric]] = None,
            **enhance_kwargs
            ) -> Tuple[Dict[str, List[float]], List[Dict[str, Union[str, np.array]]]]:
        """
        Evaluate the model by running it on a list of `audios` and calculating `metrics`.
        If `metrics` is None, will use the model's property `self.eval_metrics`.

        Input `audios` must be a list of tuples like `[(x, y, name), ...]`.
        All additional kwargs will be handed to `self.enhance()`, and can be used for instance
          to set the number of sampling steps.

        Output will be a tuple `(metric_results, output_audios)`, where
            * `output_audios` is a list of dicts, one for each input file, containing the
              original `name`, the estimate `x_hat`, the ground-truth `x`, the noisy `y`
              and the noise estimate `n = y - x`.
            * `metric_results` is a dict of lists, each key mapping the name of a metric to a list
               containing all metric values, in the same order as the input `audios`.
        """
        log.info(f"Running evaluation with enhance_kwargs: {enhance_kwargs}")
        metrics = metrics if metrics is not None else self.eval_metrics
        metric_results = []

        output_audios = []
        # Iterate over files
        for (x, y, basename) in audios:
            with torch.no_grad():
                x_hat = self.enhance(y, **enhance_kwargs)

            if x.squeeze().shape == y.squeeze().shape and x_hat.squeeze().shape == y.squeeze().shape:
                try:
                    metric_results.append(get_metrics_row(metrics, basename, x_hat.cpu(), x.cpu(), y.cpu()))
                except Exception as e:
                    log.exception("Exception occurred when calculating metrics")
            else:
                log.warning(
                    f"!!! Found mismatched shapes between x {x.shape}, y {y.shape} and x_hat {x_hat.shape}. "
                    f"Ignoring this audio ({basename}) for metric calculations!")

            output_audios.append(dict(name=basename, x_hat=x_hat, x=x, y=y))
        return pd.DataFrame(metric_results), output_audios

    # === Useful overrides of Torch/Lightning functionality ===

    # @classmethod
    # def load_from_checkpoint(
    #         cls, checkpoint_path, map_location=None, hparams_file=None, strict=None,
    #         ema=True,
    #         **kwargs):
    #     """
    #     Custom logic for loading any EnhancementModel from a checkpoint file. Uses Hydra for instantiating the model from the config stored in the checkpoint file, then loads the weights and hparams from the checkpoint file into the instance.

    #     Args:
    #         checkpoint_path: The path to the checkpoint file
    #         map_location: The map_location to be passed to `torch.load`
    #         hparams_file: *Unsupported*, will error if passed
    #         strict: Currently unused
    #         ema:
    #             Pass True to load the model with stored EMA weights (for inference).
    #             Pass False to use plain non-EMA weights (for training/finetuning).
    #     """
    #     checkpoint = torch.load(checkpoint_path, map_location=map_location)
    #     if hparams_file is not None:
    #         raise NotImplementedError("Overriding hparams from a hparams_file is unsupported!")

    #     cfg = checkpoint['hyper_parameters']
    #     model, _ = instantiate_core_objects(cfg)
    #     assert isinstance(model, EnhancementModel), f"Loading checkpoint with wrong class: {model.__class__}!"

    #     if ema:
    #         print("Loading EMA state dict...")
    #         state_dict = checkpoint['_pl_ema_state_dict']
    #     else:
    #         state_dict = checkpoint['state_dict']

    #     model.load_state_dict(state_dict)
    #     device = next((t for t in state_dict.values() if isinstance(t, torch.Tensor)), torch.tensor(0)).device
    #     return model.to(device)



# ===========  FlowModel   ===========

class FlowModel(EnhancementModel):
    """
    A flow-based generative audio enhancement model.
    """
    # https://github.com/Lightning-AI/pytorch-lightning/pull/19404
    # We use this flag so that loading from checkpoint when finetuning doesn't break
    strict_loading = False

    def __init__(
            self, flow_matcher: ConditionalFlowMatcher,
            sigma_x: Union[float, torch.Tensor], sigma_y: Union[float, torch.Tensor],
            *args, **kwargs):
        """
        Construct a new FlowModel.
        """
        super().__init__(*args, **kwargs)
        self.flow_matcher = flow_matcher
        if not callable(sigma_x):
            self.sigma_x = nn.Parameter(
                sigma_x if isinstance(sigma_x, torch.Tensor) else torch.tensor(float(sigma_x)),
                requires_grad=False)
        else:
            self.sigma_x = sigma_x
        if not callable(sigma_y):
            self.sigma_y = nn.Parameter(
                sigma_y if isinstance(sigma_y, torch.Tensor) else torch.tensor(float(sigma_y)),
                requires_grad=False)
        else:
            self.sigma_y = sigma_y

    def _loss(self, batch, batch_idx, which):
        xmu, ymu, basenames = batch
        with torch.no_grad():
            Ymu, Xmu, preprocess_info = self._preprocess(ymu, x=xmu)

        # We have to sample t ourselves as the torchcfm pkg typecasts t to Xt.dtype,
        # which is a complex dtype in our case, which we do not want for t
        t = torch.rand(Xmu.shape[0], device=Xmu.device)

        # Ys corresponds to 'x0' in Tong et al., and Xs corresponds to 'x1'.
        Ys = Ymu + self._get_noise(Ymu, self.sigma_y)
        Xs = Xmu + self._get_noise(Xmu, self.sigma_x)
        t, Xt, Ut = self.flow_matcher.sample_location_and_conditional_flow(Ys, Xs, t=t)
        Vt = self(Xt, Ymu, t)  # feed un-noised Ymu to backbone, similar to SGMSE

        # NOTE: For real-valued loss we have to use abs(...)**2 instead of just ...**2 as Vt and Ut are complex-valued!
        errs = (Vt - Ut).abs()
        # (currently unused, just an experiment) Apply error weighting. Can be per-frequency, per-time or even per-bin.
        if self.error_weighting is not None:
            # for now just implement constant weighting (not dependent on Xmu / Ymu / t / ...)
            errs = self.error_weighting.to(errs.device) * errs
        squared_errs = errs ** 2
        # Reduce to a single loss value per each sample in the batch
        per_sample_errs = squared_errs.flatten(start_dim=1).mean(dim=1)

        # NaN checks and handling
        isnans = torch.isnan(per_sample_errs)
        if torch.any(isnans):
            log.warning(f"!!!!! NaNs in at least one sample in batch {batch_idx}: {isnans}")
            per_sample_errs = per_sample_errs[~isnans]
            for idx, isnan in enumerate(isnans):
                if isnan:
                    log.warning(f"!!!!! Found NaN in batch {batch_idx} at basename {basenames[idx]}. Ignoring this sample!")
                    log.warning(f"anynan(Xmu): {torch.any(torch.isnan(Xmu[idx])).item()}, {torch.sum(torch.isnan(Xmu[idx])).item() / Xmu[idx].numel()}")
                    log.warning(f"anynan(Ymu): {torch.any(torch.isnan(Ymu[idx])).item()}, {torch.sum(torch.isnan(Ymu[idx])).item() / Ymu[idx].numel()}")
                    log.warning(f"anynan(Xs): {torch.any(torch.isnan(Xs[idx])).item()}")
                    log.warning(f"anynan(Ys): {torch.any(torch.isnan(Ys[idx])).item()}")
                    log.warning(f"anynan(Xt): {torch.any(torch.isnan(Xt[idx])).item()}")
                    log.warning(f"anynan(t): {torch.any(torch.isnan(t[idx])).item()}")
                    log.warning(f"anynan(Ut): {torch.any(torch.isnan(Ut[idx])).item()}")
                    log.warning(f"anynan(Vt): {torch.any(torch.isnan(Vt[idx])).item()}")
        if torch.all(isnans):
            # In this case we give up as it's very unlikely we'll recover from this
            raise ValueError(f"Whole batch {batch_idx} led to NaN loss values! Seems like training is broken :(")

        # Calculate the mean loss of all samples
        loss = torch.mean(per_sample_errs)
        return loss

    def forward(self, xt, y, t):
        if t.ndim == 0:
            t = t.unsqueeze(0)  # some backbones (e.g. NCSN++) cannot deal with 0-dim. t
        vt = self.backbone(xt, y, t)
        return vt

    def enhance(
            self, y, return_preprocess_info: bool = False, N: int = 50, solver: str = "euler",
            with_grad: bool = False, sigma_fac: float =1.0,
            return_traj: bool = False,  # pass to return full trajectory instead
            **kwargs):
        """
        Enhances a coded/noisy waveform y using a FlowModel

        Args:
            - y (torch.Tensor): A time-domain waveform tensor to enhance
            - return_preprocess_info: Flag to return the meta information (scaling factor, uncropping function) from the preprocessing step
            - N: the number of timesteps to use for the solver. NOTE: This is generally not equal to the NFE! e.g. midpoint has NFE=2*N, Euler has NFE=N
            - solver: The solver to use. "euler" and "midpoint" are supported -- check `NeuralODE` from `torchdyn` package for any others you may want to use
            - sigma_fac: (Experimental, unused) Factor to multiply the initial noise by
            - with_grad: Flag to enable gradients flowing. Only useful if you want to backprop through the solver.
            - return_traj: Flag to return the full feature- and time-domain trajectories of the solver, instead of just the enhanced waveform estimate. Will ignore `return_preprocess_info`.

        Returns:
            - if `return_preprocess_info=False`: just the enhanced waveform as a torch.Tensor
            - if `return_preprocess_info=True`: A 2-tuple of
                - the enhanced waveform as a torch.Tensor
                - the preprocessing info dictionary
            - if `return_traj=True`: A 2-tuple of two lists:
                - the full trajectory in feature domain (X_hats)
                - the full trajectory in time domain (x_hats)
                Note that here `return_preprocess_info` is ignored.
        """
        solver = get_solver(solver)

        grad_ctxmgr = torch.enable_grad if with_grad else torch.no_grad
        with grad_ctxmgr():
            Y, _, preprocess_info = self._preprocess(y.to(self.device))
            node_fn = (
                lambda t, Xt, *args, **kwargs: self(Xt, Y, t)
            )
            node_ = NeuralODE(node_fn, solver=solver, sensitivity="adjoint")
            initial_state = Y + sigma_fac * self._get_noise(Y, self.sigma_y)
            t_span = torch.linspace(0, 1, N+1, device=Y.device)
            traj = node_.trajectory(initial_state, t_span=t_span)
            # NFE could be accessed via: `node_.vf.nfe` here

            if return_traj:
                X_hats = traj
                x_hats = [self._postprocess(X_hat, preprocess_info=preprocess_info) for X_hat in X_hats]
                return X_hats, x_hats
            else:
                X_hat = traj[-1]
                x_hat = self._postprocess(X_hat, preprocess_info=preprocess_info)
                x_hat = x_hat.to(y.device)  # move output back to original device
        if return_preprocess_info:
            return x_hat, preprocess_info
        else:
            return x_hat

    def _get_noise(self, x, sigma):
        if isinstance(sigma, (float, int)) and sigma == 0:
            return 0.0
        elif callable(sigma):
            return sigma(x).type(x.dtype) * torch.randn_like(x)
        else:
            return (sigma * torch.randn_like(x)).type(x.dtype)


# ===========  RegressionModel   ===========

class RegressionModel(EnhancementModel):
    def __init__(self, loss_type: str = 'l2', *args, **kwargs):
        """
        Initialize a RegressionModel. Parameters:
          `loss_type`: The loss type to use. 'l2' (= squared L2 norm = MSE) by default.
            Currently, only 'l2' is implemented.
        """
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        assert self.loss_type == 'l2'

    def _loss(self, batch, batch_idx, which):
        x, y, basenames = batch
        with torch.no_grad():
            Y, X, _ = self._preprocess(y, x=x)
        t = torch.zeros(X.shape[0], device=X.device)
        Xhat = self(Y, Y, t)  # lazy passing Y twice...
        loss = torch.mean((Xhat - X).abs() ** 2)
        return loss

    # TODO rewrite API to just be forward(self, y)
    def forward(self, xt, y, t):
        if t.ndim == 0:
            t = t.unsqueeze(0)  # some backbones (e.g. NCSN++) cannot deal with 0-dim. t
        x0hat = self.backbone(xt, y, t)
        return x0hat

    def enhance(self, y, return_preprocess_info=False, **kwargs):
        with torch.no_grad():
            Y, _, preprocess_info = self._preprocess(y.to(self.device))
            t = torch.zeros(Y.shape[0], device=Y.device)
            X_hat = self(Y, Y, t)
            x_hat = self._postprocess(X_hat, preprocess_info=preprocess_info)
            x_hat = x_hat.to(y.device)  # move output back to original device
        if return_preprocess_info:
            return x_hat, preprocess_info
        else:
            return x_hat


# ===========  ScoreModel  ===========

class ScoreModel(EnhancementModel):
    def __init__(self, sde, t_eps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize SDE
        self.sde = sde
        self.t_eps = t_eps

    def _loss(self, batch, batch_idx, which):
        xs, ys, basenames = batch
        with torch.no_grad():
            Ys, Xs, _ = self._preprocess(ys, x=xs)
            ts = torch.rand(Xs.shape[0], device=Xs.device) * (self.sde.T - self.t_eps) + self.t_eps

        mean, std_batch = self.sde_mean_std(Xs, Ys, ts)
        std = self._broadcast_std(std_batch, mean)
        Zs = torch.randn_like(Ys)
        Xts = mean + Zs * std

        score_est = self(Xts, Ys, ts)
        score_gt = (-Zs / std)  # definition of (empirical) score, for Gaussian perturbation kernel
        err = score_est - score_gt
        # The following weighting stabilizes the loss, effectively resulting in the expected
        # DNN output always being standard Gaussian (z). Also cf. the definition of forward().
        err = std * err

        losses = torch.square(err.abs())
        # Sum over all channels&features, average over batch. The 0.5 is to get a classic L2 loss.
        loss = 0.5*torch.mean(torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def forward(self, xt, y, t_batch):
        """
        Outputs an approximated score of a Gaussian perturbation kernel,
        i.e., (-z/std), when given an input x_t = mu_t + std*z and z being standard Gaussian.

        For direct enhancement of a given audio tensor, use the `enhance` method instead.
        """
        # The division by sigmas is for stability: together with the loss of
        #   |self() - (-z/std)|**2 = |(-dnn()/std) - (-z/std)|**2
        # one can see that the DNN is now effectively tasked with estimating z,
        # which has the fortunate property of being standard Gaussian.
        # We do this scaling here rather than in the loss function definition, so that the
        # expected output from this forward() call is correct: it is still the score -z/std.
        std_batch = self.sde_std(t_batch)
        score = -self.backbone(xt, y, t_batch) / self._broadcast_std(std_batch, xt)
        return score

    def enhance(
        self, y,
        sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5,
        return_preprocess_info=False, **kwargs
    ):
        with torch.no_grad():
            # Prepare input (for SDE process and for DNN)
            Y, _, preprocess_info = self._preprocess(y.to(self.device))

            # Reverse sampling
            if sampler_type == "pc":
                sampler = self.get_pc_sampler(predictor, corrector, Y, N=N,
                    corrector_steps=corrector_steps, snr=snr, intermediate=False,
                    **kwargs)
            elif sampler_type == "ode":
                sampler = self.get_ode_sampler(Y, N=N, **kwargs)
            else:
                raise ValueError(f"{sampler_type} is not a valid sampler type!")

            X_hat, nfe = sampler()
            x_hat = self._postprocess(X_hat, preprocess_info=preprocess_info)
            x_hat = x_hat.to(y.device)  # move output back to original device

        if return_preprocess_info:
            return x_hat, preprocess_info
        else:
            return x_hat

    # === SDE / data helpers ===

    def sde_mean_std(self, x0, y, t_batch):
        mean, std = self.sde.marginal_prob(x0, t_batch, y)
        return mean, std

    def sde_mean(self, x0, y, t_batch):
        return self.sde._mean(x0, y, t_batch)

    def sde_std(self, t_batch):
        return self.sde._std(t_batch)

    def _broadcast_std(self, std, x):
        return repeat(std, 'b -> b c t f', **parse_shape(x, "b c t f"))

    # === Functions concerned with inference ===

    def get_pc_sampler(self, predictor_name, corrector_name, ys, N=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N
        kwargs = {"eps": self.t_eps, **kwargs}
        return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=ys, **kwargs)

    def get_ode_sampler(self, ys, N=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N
        kwargs = {"eps": self.t_eps, **kwargs}
        return sampling.get_ode_sampler(sde, self, y=ys, **kwargs)
