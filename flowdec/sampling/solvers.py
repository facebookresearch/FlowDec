# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Some custom solvers for torchdyn, particularly order-2 Heun.
"""

import torch
from torchdyn.numerics.solvers.templates import DiffEqSolver


class Heun2(DiffEqSolver):
    """
    Explicit Heun ODE stepper, order 2.

    Seems to work consistently worse than Midpoint for our task (introduces residual noise artifacts).
    Could be due to evaluation at t=1 in the last step.
    """
    def __init__(self, dtype=torch.float32):
        """Explicit Heun ODE stepper, order 2"""
        super().__init__(order=2)
        self.dtype = dtype
        self.stepping_class = 'fixed'

    def step(self, f, x, t, dt, k1=None, args=None):
        if k1 == None: k1 = f(t, x)
        x_pred = x + dt * k1
        f_pred = f(t + dt, x_pred)
        x_sol = x + dt*0.5 * (k1 + f_pred)
        return None, x_sol, None


class Heun2_EulerLast(DiffEqSolver):
    """
    Explicit Heun ODE stepper, order 2.
    Uses Euler in the "last" step (i.e. when t+dt == 1.0) to avoid evaluation at t=1.0.
    """
    def __init__(self, dtype=torch.float32):
        super().__init__(order=2)
        self.dtype = dtype
        self.stepping_class = 'fixed'

    def step(self, f, x, t, dt, k1=None, args=None):
        if k1 == None: k1 = f(t, x)
        x_pred = x + dt * k1

        is_last_step = torch.isclose(t+dt, torch.ones_like(t))
        if torch.all(is_last_step):
            x_sol = x_pred
        else:
            f_pred = f(t + dt, x_pred)
            x_corr = x + dt*0.5 * (k1 + f_pred)
            x_sol = torch.where(is_last_step, x_pred, x_corr)
        return None, x_sol, None



CUSTOM_SOLVERS = {'heun2': Heun2, 'heun2_eulerlast': Heun2_EulerLast}


def get_solver(name, *args, **kwargs):
    if name in CUSTOM_SOLVERS:
        return CUSTOM_SOLVERS[name](*args, **kwargs)
    else:
        return name  # let torchdyn handle this string
