# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Tuple
import copy

from hydra.utils import instantiate
import omegaconf
import pytorch_lightning as pl


def instantiate_core_objects(
        cfg: Union[omegaconf.DictConfig, dict]
    ) -> Tuple[pl.LightningModule, pl.LightningDataModule]:
    """
    Instantiates 'core objects' from a Hydra config: The optimizer, datamodule, and model.

    Returns a 2-tuple of instances `(model, datamodule)`.

    NOTE: the optimizer is not returned since it is passed to `model` in the form of a callable, which will be ru
    by the model `prepare_optimizers` methods to actually instantiate the optimizer with the model parameters.
    """
    optimizer_init = instantiate(cfg.optimizer)  # should be a partial, expecting (parameters, lr=...)
    datamodule = instantiate(cfg.datamodule)
    model = instantiate(
        cfg.model, optimizer_init=optimizer_init, datamodule=datamodule,
        # Pass as YAML so it's not instantiated (but resolved). We'll unmarshal it in the model constructor
        full_config=omegaconf.OmegaConf.to_yaml(get_loggable_cfg(cfg))
    )
    return model, datamodule


def get_loggable_cfg(cfg: Union[omegaconf.DictConfig, dict]) -> dict:
    """
    Converts an instantiated Hydra config into a dictionary we can use for logging and for storing in checkpoints,
    so this config can later be loaded from the checkpoint and used for model re-instantiation.

    Additionally, adds a 'wb_listfix' key that contains a copy of the entire config, but with all lists-of-dictionaries replaced by a dictionary-of-dictionaries that maps the indices as integers 0,...n-1 to the list entries.
    This is a workaround for the behavior of W&B that it does not let us access config entries within lists for
    filtering runs etc.
    """
    cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # This block of code is a workaround for an annoying behavior of W&B: We cannot access properties
    # of list entries inside any config for things like run filtering and parameter importance analysis.
    # So we convert all those lists anywhere in our config into a dict mapping list indices to the entries.
    # This is a format that W&B likes better for some reason.
    #
    # We store the converted config in an unused key 'wb_listfix' on the original config
    # in order to not mess up model instantiation, and to keep the rest of the config as-is.
    cfg_copy = copy.deepcopy(cfg)
    if 'wb_listfix' in cfg:
        del cfg['wb_listfix']
    cfg['wb_listfix'] = convert_list_of_dicts(cfg_copy)
    return cfg


def _convert_list_of_dicts(config):
    """
    Helper for get_loggable_cfg.
    """
    for key, value in list(config.items()):
        if isinstance(value, list) and any(isinstance(item, dict) for item in value):
            # Convert list of dictionaries to dictionary of dictionaries using index as key
            config[key] = {str(i): item for i, item in enumerate(value)}
        if isinstance(value, dict):
            # Recursively apply the conversion to nested dictionaries
            config[key] = convert_list_of_dicts(value)


def convert_list_of_dicts(config):
    """
    Helper for get_loggable_cfg. Recursively convert all lists of dictionaries in the config to a dictionary
    of dictionaries.
    """
    config_updated = copy.deepcopy(config)
    _convert_list_of_dicts(config_updated)
    return config_updated