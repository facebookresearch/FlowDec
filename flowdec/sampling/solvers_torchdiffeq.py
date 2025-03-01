# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom solvers for torchdiffeq.
Importing this file will modify internals of torchdiffeq on runtime, so use with caution.
"""

import warnings
warnings.warn(f"Importing this file {__file__} modifies package internals of the `torchdiffeq` package. "
              "Make sure you have a compatible version, e.g. torchdiffeq==0.2.3 !")

# ugly hack importing third-party internals :/
from torchdiffeq._impl.misc import Perturb
from torchdiffeq._impl.odeint import SOLVERS
from torchdiffeq._impl.solvers import FixedGridODESolver

class Heun2(FixedGridODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        t_pred = t0 + dt
        y_pred = y0 + f0 * dt
        f_pred = func(t_pred, y_pred)
        return dt*0.5 * (f0 + f_pred), f0


# ugly hack modifying a third-party pkg :/
SOLVERS['heun2'] = Heun2
