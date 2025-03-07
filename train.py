
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import logging
import signal
from typing import List, Any

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import omegaconf
import torch
import torch.distributed
import torchinfo
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from flowdec.util.logging import log
from flowdec.util.hydra import get_loggable_cfg, instantiate_core_objects


"""
Directories that will be ignored when logging code with W&B.
"""
IGNORED_CODE_DIRS = ['.ipynb_checkpoints', 'wandb', '.wandb']

def IS_RANK_ZERO():
    """
    Returns True if the current process is rank-0, False otherwise.
    """
    return os.getenv("LOCAL_RANK", '0') == '0'


@hydra.main(config_path="./config", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    # Set up console logger (just set loglevel here given cfg)
    log.setLevel(getattr(logging, cfg.loglevel.upper()))

    # Check for either manual resume_from_checkpoint option,
    # or try finding the checkpoint based on SLURM job ID and continue from there
    resume_ckpt = getattr(cfg, 'resume_from_checkpoint', None)
    if resume_ckpt is None and SLURMEnvironment.detect():
        log.info("No explicit resume_from_checkpoint -- trying to find a SLURM ckpt...")
        resume_ckpt = find_latest_slurm_ckpt(cfg)
        cfg.resume_from_checkpoint = resume_ckpt

    if cfg.get('finetune'):
        if resume_ckpt is None or not os.path.isfile(resume_ckpt):
            raise ValueError(f"config.finetune is set to True but {resume_ckpt=} does not exist!")
        log.info(f"Finetuning from {resume_ckpt} with a new run ID...")
    else:
        if resume_ckpt is not None:
            log.info(f"Resuming from checkpoint: {resume_ckpt}")
            if cfg.get('force_new_run'):
                log.info("Forcing new run..")
            else:
                log.info("Trying to find previous run ID...")
                run_id = torch.load(resume_ckpt, map_location='cpu').get('hyper_parameters', {}).get('run_id', None)
                cfg.run_id = run_id if not callable(run_id) else run_id()
            log.info(f"Found previous run ID: {cfg.run_id}")
        log.info("Did not find a checkpoint file to resume from. Starting from scratch.")

    if IS_RANK_ZERO():
        # Set up run logger and retrieve run ID from it
        config_name = HydraConfig.get()['job']['config_name']
        logger, run_id = instantiate_run_logger(cfg, config_name=config_name)
        log.info(f"=== Run ID: {run_id} === ")

    # Set up core objects: model [LightningModule], data module [LightningDataModule]
    model, datamodule = instantiate_core_objects(cfg)
    log.info(f"=== Data module: ===\n{datamodule}")
    log.info(f"=== Model summary by torchinfo: ===\n{torchinfo.summary(model, verbose=0)}")

    # Set up callbacks
    callbacks = instantiate_callbacks(cfg, run_id) if IS_RANK_ZERO() else None

    # Set up SLURM environment (if detected)
    plugins = []
    if SLURMEnvironment.detect():  # check if we are running in a SLURM environment
        set_up_slurm(cfg, run_id, plugins)
        log.info("Successfully set up SLURM.")

    # Set up pl.Trainer instance
    log.info("Instantiating pl.Trainer.")
    trainer = pl.Trainer(
        logger=logger if IS_RANK_ZERO() else False,
        callbacks=callbacks,
        plugins=plugins,
        # This option is a workaround for annoying SLURMEnvironment behavior, which uses the trainer default_root_dir
        # for storing checkpoints when the job is pre-empted (which is the code dir by default)
        default_root_dir=os.path.join(logger.save_dir, 'cont', logger.version) if IS_RANK_ZERO() else None,
        **cfg.trainer_options
    )
    log.info("Done instantiating pl.Trainer.")

    # Global matmul precision setting
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # Train model
    log.info("Starting trainer.fit() loop...")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=cfg.get('resume_from_checkpoint', None),
    )
    log.info("Finished trainer.fit() loop!")

    # This part is only needed for using Hydra multirun / sweeping, e.g. for grid searches
    # see https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/
    optimized_metrics = cfg.get("optimized_metrics")
    log.info(f"Found optimized_metrics: {optimized_metrics}.")
    if optimized_metrics is not None:
        log.info("Returning final metrics from completed trainer.fit run.")
        final_metrics = {**trainer.logged_metrics, **trainer.callback_metrics}
        return [final_metrics[m] for m in optimized_metrics]


def get_checkpoints_dir(cfg: omegaconf.DictConfig):
    """
    Gets the setting for the checkpoints directory from a instantiated config.
    """
    return cfg.dirs.checkpoint_dir


def get_slurm_job_id():
    """
    Gets the SLURM job ID (if present).
    """
    return str(SLURMEnvironment.job_id())


def find_latest_slurm_ckpt(cfg: omegaconf.DictConfig):
    """
    Finds the latest checkpoint from the current SLURM job ID. Used for auto-resuming after preemption.
    """
    ckpts_dir = get_checkpoints_dir(cfg)
    slurm_job_id = get_slurm_job_id()
    ckpt_dir = os.path.join(ckpts_dir, slurm_job_id)
    ckpt_files = glob.glob(os.path.join(ckpt_dir, '*last*.ckpt'))
    if not ckpt_files:  # None found
        return None
    last_ckpt_file = list(sorted(
        ckpt_files, key=lambda f: int(f.split('epoch=')[1].split('-')[0])
    ))[-1]
    return last_ckpt_file


def set_up_slurm(cfg: omegaconf.DictConfig, run_id: str, plugins: List[Any]):
    """
    Sets up SLURM for this training run. More specifically:
      * adds an appropriate SLURMEnvironment() plugin instance to the passed `plugins` list,
        which is useful for when the job is preempted / requeued
      * symlinks a checkpoint directory from the SLURM job ID to the actual run ID (`run_id`)

    Args:
        - cfg: an instantiated Hydra config
        - run_id: a unique ID for this training run (usually auto-determined by the W&B logger)
        - plugins: List of PyTorch Lightning plugins that the configured SLURMEnvironment() instance
          will be appended to.
    """
    # Set up auto-requeue on SLURM pre-emption,
    # see https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
    plugins.append(SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGUSR1))
    if 'SLURM_RESTART_COUNT' in os.environ:
        log.info(f"This job has been pre-empted {os.environ['SLURM_RESTART_COUNT']} times.")

    # Set up symlink from SLURM job id to logger run ID, in the checkpoints dir.
    # Useful for auto-resuming based on SLURM job id when pre-empted / rescheduled.
    ckpts_dir = get_checkpoints_dir(cfg)
    symlink_target = os.path.join(ckpts_dir, run_id)
    symlink_name = os.path.join(ckpts_dir, get_slurm_job_id())
    if not (os.path.islink(symlink_name) or os.path.exists(symlink_name)):
        try:
            os.symlink(symlink_target, symlink_name, target_is_directory=True)
        except Exception as e:
            log.warn(f"Failed setting up symlink due to exception: {e}")
        log.info(f"Set up a checkpoint dir symlink {symlink_name} pointing to {symlink_target}.")
    else:
        log.info(f"Tried to set up a checkpoint dir symlink but {symlink_name} already exists.")


def instantiate_run_logger(cfg: omegaconf.DictConfig, config_name: str):
    """
    Fully instantiates the logger instance for this run from the given config `cfg`.
    Supports `WandbLogger` and `TensorBoardLogger` from PyTorch Lightning.

    Args:
        - cfg: an instantiated Hydra config
        - config_name: The unique name of the config used for config instantiation, e.g. "13_flowdec_v1" when
          the config file `config/13_flowdec_v1.yaml` is present. Will be logged as a run hyperparameter for
          future reference.
    """
    logger_kwargs = {}
    prev_run_id = getattr(cfg, 'run_id', None)
    if prev_run_id is not None:
        log.info(f"Found previous run_id: {prev_run_id}! Passing that for logger instantiation...")
        logger_kwargs.update({'version': prev_run_id})  # 'version' should work for both W&B and TensorBoard

    logged_cfg = {**get_loggable_cfg(cfg), 'config_name': config_name}

    # Set up run logger
    logger = instantiate(cfg.logger, **logger_kwargs)
    if isinstance(logger, WandbLogger):
        run = logger.experiment
        run_id = run.id

        # Store run ID in the config, so it's saved in the checkpoint, and can be used to continue
        if cfg.run_id is None:
            cfg.run_id = run_id

        basedir = os.path.dirname(__file__)
        run.log_code(
            basedir,
            include_fn=lambda path: path.endswith(".py"),
            exclude_fn=lambda path: any(
                os.path.relpath(path, basedir).startswith(pat) for pat in IGNORED_CODE_DIRS
            )
        )
        # Nicer format of hparam/config logging for W&B, since doing it via
        # model.save_hyperparameters() would result in all values being stringified
        run.config.update(logged_cfg, allow_val_change=True)
    elif isinstance(logger, TensorBoardLogger):
        run_id = os.path.basename(logger.log_dir)
        # Store run ID in the config, so it's saved in the checkpoint, and can be used to continue
        if cfg.run_id is None:
            cfg.run_id = run_id

        logger.log_hyperparams(logged_cfg)
    else:
        raise NotImplementedError("Only WandbLogger and TensorBoardLogger are implemented as loggers!")
    return logger, run_id


def instantiate_callbacks(cfg: omegaconf.DictConfig, run_id: str):
    """
    Fully instantiates the callbacks from the config `cfg`. Specifically:

    * updates any ModelCheckpoint callback with an output directory corresponding to the `run_id`.

    Args:
        - cfg: an instantiated Hydra config
        - run_id: The unique ID of this training run (usually auto-determined by W&B)
    """
    callbacks = instantiate(cfg.callbacks)
    # We programmatically update any ModelCheckpoint dirpaths depending on the run ID,
    # so that we save checkpoints to a neat unique subdirectory.
    # TODO is there a better way?
    for callback in callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            callback.dirpath = os.path.join(callback.dirpath, run_id)
            log.info(f"Updated dirpath for {callback} with run_id={run_id}. Now: {callback.dirpath}")
    return callbacks


if __name__ == "__main__":
    main()
