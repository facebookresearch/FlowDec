# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Callable


class Registry:
    """
    A kind of outdated class-management utility used by the old ScoreDec / SGMSE+ code,
    in order to register subclasses and provide simple access to them by a unique string key.

    Only used for managing ScoreDec's `SDE`, `Predictor` and `Corrector` instances.
    """
    def __init__(self, managed_thing: str):
        """
        Create a new registry.

        Args:
            managed_thing: A string describing what type of thing is managed by this registry. Will be used for
                warnings and errors, so it's a good idea to keep this string globally unique and easily understood.
        """
        self.managed_thing = managed_thing
        self._registry = {}

    def register(self, name: str) -> Callable:
        def inner_wrapper(wrapped_class) -> Callable:
            if name in self._registry:
                warnings.warn(f"{self.managed_thing} with name '{name}' doubly registered, old class will be replaced.")
            self._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    def get_by_name(self, name: str):
        """Get a managed thing by name."""
        if name in self._registry:
            return self._registry[name]
        else:
            raise ValueError(f"{self.managed_thing} with name '{name}' unknown.")

    def get_all_names(self):
        """Get the list of things' names registered to this registry."""
        return list(self._registry.keys())
