# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# The code is copied and adapted from https://github.com/BioinfoMachineLearning/bio-diffusion/blob/e4bad15139815e562a27fb94dab0c31907522bc5/src/utils/__init__.py
# Original code is under a MIT license: https://github.com/BioinfoMachineLearning/bio-diffusion/blob/1cfc969193ee9f32d5300c63726b33a2a3b071d9/LICENSE

import torch

import pytorch_lightning as pl

from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any, Dict, List, Optional

from flowdec.util.logging import log

try:
    import amp_C
    apex_available = True
except Exception:
    apex_available = False


class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).
    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters under the dictionary key `ema`.
    Args:
        decay: The exponential decay used when calculating the moving average. Has to be between 0-1.
        apply_ema_every_n_steps: Apply EMA every n global steps.
        start_step: Start applying EMA from ``start_step`` global step onwards.
        save_ema_weights_in_callback_state: Enable saving EMA weights in callback state.
        evaluate_ema_weights_instead: Validate the EMA weights instead of the original weights.
            Note this means that when saving the model, the validation metrics are calculated with the EMA weights.

    Adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py
    """

    def __init__(
        self,
        decay: float,
        apply_ema_every_n_steps: int = 1,
        start_step: int = 0,
        evaluate_with_ema_weights: bool = True,
    ):
        if not apex_available:
            rank_zero_warn(
                "EMA has better performance when Apex is installed: https://github.com/NVIDIA/apex#installation."
            )
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self._ema_model_weights: Optional[List[torch.Tensor]] = None
        self._overflow_buf: Optional[torch.Tensor] = None
        self._cur_step: Optional[int] = None
        self._weights_buffer: Optional[List[torch.Tensor]] = None
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.evaluate_ema_weights_instead = evaluate_with_ema_weights
        self.decay = decay

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        log.info("Creating EMA weights copy.")
        if self._ema_model_weights is None:
            self._ema_model_weights = [p.detach().clone() for p in pl_module.state_dict().values()]
        # ensure that all the weights are on the correct device
        self._ema_model_weights = [p.to(pl_module.device) for p in self._ema_model_weights]
        self._overflow_buf = torch.IntTensor([0]).to(pl_module.device)

    def ema(self, pl_module: "pl.LightningModule") -> None:
        if apex_available and pl_module.device.type == "cuda":
            return self.apply_multi_tensor_ema(pl_module)
        return self.apply_ema(pl_module)

    def apply_multi_tensor_ema(self, pl_module: "pl.LightningModule") -> None:
        model_weights = list(pl_module.state_dict().values())
        amp_C.multi_tensor_axpby(
            65536,
            self._overflow_buf,
            [self._ema_model_weights, model_weights, self._ema_model_weights],
            self.decay,
            1 - self.decay,
            -1,
        )

    def apply_ema(self, pl_module: "pl.LightningModule") -> None:
        log.debug("Applying EMA weight update")
        for orig_weight, ema_weight in zip(list(pl_module.state_dict().values()), self._ema_model_weights):
            if ema_weight.data.dtype != torch.long and orig_weight.data.dtype != torch.long:
                # ensure that non-trainable parameters (e.g., feature distributions) are not included in EMA weight averaging
                diff = ema_weight.data - orig_weight.data
                diff.mul_(1.0 - self.decay)
                ema_weight.sub_(diff)

    def should_apply_ema(self, step: int) -> bool:
        return step != self._cur_step and step >= self.start_step and step % self.apply_ema_every_n_steps == 0

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.should_apply_ema(trainer.global_step):
            self._cur_step = trainer.global_step
            self.ema(pl_module)

    def state_dict(self) -> Dict[str, Any]:
        return dict(cur_step=self._cur_step, ema_weights=self._ema_model_weights)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._cur_step = state_dict["cur_step"]
        self._ema_model_weights = state_dict.get("ema_weights")

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint_callback = trainer.checkpoint_callback

        if trainer.ckpt_path and checkpoint_callback is not None:
            if "_pl_ema_state_dict" in checkpoint:
                self._ema_model_weights = checkpoint["_pl_ema_state_dict"].values()
                log.info("EMA weights have been loaded successfully. Continuing training with saved EMA weights.")
            else:
                log.warn(
                    "We were unable to find the associated EMA weights when re-loading, "
                    "training will start with new EMA weights.",
                )

    def get_ema_state_dict(self, pl_module: "pl.LightningModule") -> Dict[str, Any]:
        ema_state_dict = {k: v for k, v in zip(pl_module.state_dict().keys(), self._ema_model_weights)}
        return ema_state_dict

    def replace_model_weights(self, pl_module: "pl.LightningModule") -> None:
        log.debug("Replacing model weights with EMA weights.")
        self._weights_buffer = [p.detach().clone().to("cpu") for p in pl_module.state_dict().values()]
        ema_state_dict = self.get_ema_state_dict(pl_module)
        pl_module.load_state_dict(ema_state_dict)

    def restore_original_weights(self, pl_module: "pl.LightningModule") -> None:
        log.debug("Replacing EMA weights with original model weights.")
        state_dict = pl_module.state_dict()
        new_state_dict = {k: v for k, v in zip(state_dict.keys(), self._weights_buffer)}
        pl_module.load_state_dict(new_state_dict)
        del self._weights_buffer

    # __enter__ and __exit__ allow us to use the EMA as a context manager with a clean API
    def __enter__(self, pl_module):
        self._ctxmgr_affected_pl_module = pl_module
        self.replace_model_weights(pl_module)

    def __exit__(self):
        if getattr(self, '_ctxmgr_affected_pl_module', None) is None:
            raise ValueError("Trying to exit with(EMA) block but the affected lighting module was not saved?!")
        self.restore_original_weights(self._affected_pl_module)
        del self._ctxmgr_affected_pl_module

    @property
    def ema_initialized(self) -> bool:
        return self._ema_model_weights is not None

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)


class EMAModelCheckpoint(ModelCheckpoint):
    """
    Light wrapper around Lightning's `ModelCheckpoint` to, upon request, save an EMA copy of the model as well.

    Adapted from: https://github.com/NVIDIA/NeMo/blob/be0804f61e82dd0f63da7f9fe8a4d8388e330b18/nemo/utils/exp_manager.py#L744
    """

    def __init__(self, **kwargs):
        # call the parent class constructor with the provided kwargs
        super().__init__(**kwargs)

    def _get_ema_callback(self, trainer: "pl.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        # FIXME this accesses some Lightning internals and is hacky. But I don't know how else to
        # modify the checkpoint dict to be stored *before storing it*, and I also don't want to store
        # the EMA weights in a *separate* checkpoint file somewhere. So here we are.

        # This gets the original checkpoint as intended by pl.Trainer:
        checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
        ## BEGIN this is added over just trainer.save_checkpoint()
        ema_callback = self._get_ema_callback(trainer)
        ema_state_dict = ema_callback.get_ema_state_dict(trainer.lightning_module)
        checkpoint["_pl_ema_state_dict"] = ema_state_dict
        ## END this is added over just trainer.save_checkpoint()
        # This saves the combined checkpoint to the intended filepath and employs a barrier
        trainer.strategy.save_checkpoint(checkpoint, filepath)
        trainer.strategy.barrier("Trainer.save_checkpoint")

    # See https://github.com/Lightning-AI/pytorch-lightning/issues/12724#issuecomment-2046532723
    def _save_topk_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict[str, torch.Tensor]) -> None:
        if self.save_top_k == 0:
            return
        # validate metric
        if self.monitor is not None:
            if self.monitor not in monitor_candidates:
                m = (
                    f"`EMAModelCheckpoint(monitor={self.monitor!r})` could not find the monitored key in the returned"
                    f" metrics: {list(monitor_candidates)}."
                    f" HINT: Did you call `log({self.monitor!r}, value)` in the `LightningModule`? "
                    f" ...Continuing in any case (overridden behavior)"
                )
                log.warn(m)
            self._save_monitor_checkpoint(trainer, monitor_candidates)
        else:
            self._save_none_monitor_checkpoint(trainer, monitor_candidates)
