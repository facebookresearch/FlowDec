# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
import warnings

import numpy as np
import torch

from flowdec.util.registry import Registry
from flowdec.util.other import batch_broadcast

import scipy.special as sc


SDERegistry = Registry("SDE")


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass

    def discretize(self, x, t, *args):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t, *args)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        std_fn = oself._std
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, t, *args)
                total_drift, diffusion = rsde_parts["total_drift"], rsde_parts["diffusion"]
                return total_drift, diffusion

            def rsde_parts(self, x, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, t, *args)
                score = score_model(x, *args, t)
                score_drift = -sde_diffusion[:, None, None, None]**2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(sde_diffusion) if self.probability_flow else sde_diffusion
                total_drift = sde_drift + score_drift
                return {
                    'total_drift': total_drift, 'diffusion': diffusion, 'sde_drift': sde_drift,
                    'sde_diffusion': sde_diffusion, 'score_drift': score_drift, 'score': score,
                }

            def discretize(self, x, t, *args):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, *args)
                rev_f = f - G[:, None, None, None] ** 2 * score_model(x, *args, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass


@SDERegistry.register("ouve")
class OUVESDE(SDE):
    def __init__(self, theta, sigma_min, sigma_max, N=1000):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = -theta (y-x) dt + sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            theta: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.theta = theta
        #print(f"THETA: {theta}")
        #print(f"SIGMA: {sigma_min} - {sigma_max}")
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N

    def copy(self):
        return OUVESDE(self.theta, self.sigma_min, self.sigma_max, N=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, t, y):
        drift = self.theta * (y - x)
        # the sqrt(2*logsig) factor is required here so that logsig does not in the end affect the perturbation kernel
        # standard deviation. this can be understood from solving the integral of [exp(2s) * g(s)^2] from s=0 to t
        # with g(t) = sigma(t) as defined here, and seeing that `logsig` remains in the integral solution
        # unless this sqrt(2*logsig) factor is included.
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return drift, diffusion

    def _mean(self, x0, t, y):
        theta = self.theta
        exp_interp = torch.exp(-theta * t)[:, None, None, None]
        return exp_interp * x0 + (1 - exp_interp) * y

    def _std(self, t):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig
        # could maybe replace the two torch.exp(... * t) terms here by cached values **t
        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2 * theta * t)
                * (torch.exp(2 * (theta + logsig) * t) - 1)
                * logsig
            )
            /
            (theta + logsig)
        )

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        x_T = y + torch.randn_like(y) * std[:, None, None, None]
        return x_T


@SDERegistry.register("ouvp")
class OUVPSDE(SDE):
    # !!! We do not utilize this SDE in our works due to observed instabilities around t=0.2. !!!
    def __init__(self, beta_min, beta_max, stiffness=1, N=1000, **ignored_kwargs):
        """
        !!! We do not utilize this SDE in our works due to observed instabilities around t=0.2. !!!

        Construct an Ornstein-Uhlenbeck Variance Preserving SDE:

        dx = -1/2 * beta(t) * stiffness * (y-x) dt + sqrt(beta(t)) * dw

        with

        beta(t) = beta_min + t(beta_max - beta_min)

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        Args:
            beta_min: smallest sigma.
            beta_max: largest sigma.
            stiffness: stiffness factor of the drift. 1 by default.
            N: number of discretization steps
        """
        super().__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.stiffness = stiffness
        self.N = N

    def copy(self):
        return OUVPSDE(self.beta_min, self.beta_max, self.stiffness, N=self.N)

    @property
    def T(self):
        return 1

    def _beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, x, t, y):
        drift = 0.5 * self.stiffness * batch_broadcast(self._beta(t), y) * (y - x)
        diffusion = torch.sqrt(self._beta(t))
        return drift, diffusion

    def _mean(self, x0, t, y):
        b0, b1, s = self.beta_min, self.beta_max, self.stiffness
        x0y_fac = torch.exp(-0.25 * s * t * (t * (b1-b0) + 2 * b0))[:, None, None, None]
        return y + x0y_fac * (x0 - y)

    def _std(self, t):
        b0, b1, s = self.beta_min, self.beta_max, self.stiffness
        return (1 - torch.exp(-0.5 * s * t * (t * (b1-b0) + 2 * b0))) / s

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        x_T = y + torch.randn_like(y) * std[:, None, None, None]
        return x_T


@SDERegistry.register("bbed")
class BBEDSDE(SDE):
    def __init__(self, c, k, N=1000, **ignored_kwargs):
        """Construct a Brownian Bridge with Exponential Diffusion Coefficient SDE. See Lay et al. 2024 (ICASSP)
        """
        super().__init__(N)
        print(f"C: {c}")
        print(f"K: {k}")
        self.c = c
        self.k = k
        self.N = N
        self.c2 = self.c ** 2
        self.logk = np.log(k)
        self.kk2logk = 2 * self.k * self.k * self.logk
        self.ei_logk = sc.expi(-2 * self.logk)

    def copy(self):
        return BBEDSDE(self.c, self.k, N=self.N)

    @property
    def T(self):
        return 0.999

    def sde(self, x, t, y):
        drift = (y - x) / (1-t)
        diffusion = self.c * (self.k ** t)
        return drift, diffusion

    def _mean(self, x0, t, y):
        t = t[:, None, None, None]
        return (1-t) * x0 + t * y

    def _std(self, t):
        t1 = (1-t)
        e = self._e(t) # Eis
        var = (self.k ** (2*t) - t1) + (self.kk2logk * t1 * e)
        return torch.sqrt((t1 * self.c2 * var))

    def _e(self, t):
        t = t.cpu().numpy()
        e = sc.expi(2 * (t-1) * self.logk) - self.ei_logk
        return torch.from_numpy(e).cuda()

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device)*self.T)
        x_T = y + torch.randn_like(y) * std[:, None, None, None]
        return x_T
