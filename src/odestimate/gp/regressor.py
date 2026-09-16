# Implements gaussian processes
# GP hyperparameters are fitted via maximum likelihood estimation (MLE)
from typing import Literal

import numpy as np
import torch
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from odestimate.gp.kernels import RBFWithNoise

type Array = np.ndarray | torch.Tensor  # type alias for convenience


def _as_numpy(a: Array) -> np.ndarray:
    """Converts a torch tensor to numpy (detaching/moving to CPU first); leaves a numpy array as
    is - `GaussianProcessRegressor` itself only accepts numpy, so every entry point below runs
    input through this first."""
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a, dtype=float)


class GP:
    def __init__(self, t_obs: Array, y_obs: Array, kernel: "Literal['rbf'] | Kernel" = "rbf", **kwargs):
        """
        Fits a Gaussian process to the observed data (t_obs, y_obs) using sklearn's
        GaussianProcessRegressor with the specified kernel.
        The hyperparameters of the GP are optimized via maximum likelihood estimation (MLE).

        Parameters
        ------------
        t_obs : Array, shape (n,)
            Observed time points.
        y_obs : Array, shape (n,)
            Observed values at `t_obs` (noisy - this is raw data, not a smoothed trajectory).
        kernel : "rbf" or a `sklearn.gaussian_process.kernels.Kernel` instance
            `"rbf"` (default): `RBFWithNoise()` (see `odestimate.gp.kernels`) - RBF covariance plus
            additive observation noise, all three hyperparameters (`length_scale`, `signal_var`,
            `noise_var`) fit jointly. 
        **kwargs : dict
            Additional keyword arguments forwarded to `GaussianProcessRegressor` (e.g.
            `n_restarts_optimizer`, `normalize_y`, `alpha` for a fixed extra nugget).
        """
        if kernel == "rbf":
            kernel = RBFWithNoise()
        elif not isinstance(kernel, Kernel):
            # Covers BOTH "not a Kernel instance at all" (wrong type) AND "some other string"
            # (e.g. "matern") - any string other than "rbf" fails `isinstance(kernel, Kernel)` too,
            # so a single branch handles both cases; no separate `elif type(kernel) == str` needed
            # (that would be unreachable - a bad string is already caught here).
            raise TypeError(
                f"kernel must be 'rbf' or a sklearn Kernel instance (not a class, another string, "
                f"or a bare callable), got {kernel!r}."
            )

        t_obs = _as_numpy(t_obs).reshape(-1, 1)
        y_obs = _as_numpy(y_obs)

        gpr = GaussianProcessRegressor(kernel=kernel, copy_X_train=False, **kwargs)
        gpr.fit(t_obs, y_obs)
        self.gpr = gpr

    def __call__(
        self, t: Array, return_std: bool = True, return_cov: bool = False,
    ) -> "tuple[np.ndarray, np.ndarray | None, np.ndarray | None]":
        """
        Predicts the mean (always) and, on request, the standard deviation and/or covariance of
        the GP's LATENT function `x_hat(t)` at the query points `t`.

        Deliberately NOT `self.gpr.predict(t, return_std=..., return_cov=...)`: sklearn's own
        `predict` computes the posterior of a NEW NOISY OBSERVATION, not of the latent function -
        internally it calls `kernel_(X)` (`Y=None`) on the QUERY points, and for a kernel like ours
        that adds `noise_var` on the `Y=None` diagonal (see `RBFWithNoise`'s own docstring), this
        silently ADDS `noise_var` into `std`/`cov` rather than merely flooring at it. That would be
        a DIFFERENT quantity than `_FittedGP.var`/`window_var` in the from-scratch implementation
        (and everywhere else in this project that reasons about GP uncertainty) - there, `noise_var`
        is only ever a floor on the latent posterior variance, never added into it. This method
        reproduces THAT convention instead: the noise-free prior self-covariance is obtained via
        `kernel(t, t)` (`Y` given EXPLICITLY, even though `Y is t`) rather than `kernel(t)`
        (`Y=None`) - `RBFWithNoise.__call__` dispatches purely on whether `Y` was omitted, not on
        whether the two arrays happen to hold equal values, so this reliably takes the no-noise
        branch. `noise_var` is then only used as a FLOOR on `std`, matching `_FittedGP.var`.

        Parameters
        ------------
        t : Array, shape (m,)
            Time points at which to predict the GP.
        return_std : bool, optional
            Default `True`. If `False` and `return_cov=False`, only `mean` is computed - no
            triangular solve at all (see Time complexity below).
        return_cov : bool, optional
            Default `False`. Requesting this is genuinely more expensive than `return_std` alone
            (see Time complexity) - ask for it only when you need the OFF-diagonal structure (e.g.
            correlated windows), not just a per-point uncertainty.

        Returns
        -------
        mean : np.ndarray, shape (m,)
        std : np.ndarray, shape (m,) or None
            `None` if `return_std=False` and `return_cov=False`. Floored at `noise_var`.
        cov : np.ndarray, shape (m, m) or None
            `None` unless `return_cov=True`.

        Time complexity
        ----------------
        `mean` alone: `O(m * n_obs)`, no triangular solve (`alpha_` is already fully baked from
        `fit()`). `return_std` (no `return_cov`): adds the `O(m * n_obs^2)` solve, but avoids ever
        forming an `(m,m)` matrix - the noise-free prior variance is a single scalar for this
        (stationary) kernel, read off via ONE `O(1)` self-covariance call and broadcast, not
        `diag(kernel(t,t))` (`O(m^2)`). `return_cov`: `O(m^2 * n_obs)` (the `v.T @ v` term) plus
        `O(m^2)` memory for `cov` itself - by far the most expensive of the three, and the reason
        this method (unlike a plain `.predict()`) lets you skip it when you only need `std`.
        """
        t = _as_numpy(t).reshape(-1, 1)
        kernel = self.gpr.kernel_
        X_train = self.gpr.X_train_
        alpha = self.gpr.alpha_
        y_train_std = getattr(self.gpr, "_y_train_std", 1.0)
        y_train_mean = getattr(self.gpr, "_y_train_mean", 0.0)

        k_star = kernel(t, X_train)  # (m, n_obs), noise-free (Y given explicitly)
        mean = k_star @ alpha * y_train_std + y_train_mean

        if not return_std and not return_cov:
            return mean, None, None

        v = solve_triangular(self.gpr.L_, k_star.T, lower=True)  # (n_obs, m)

        if return_cov:
            prior_cov = kernel(t, t)  # noise-free, (m, m) - see docstring
            cov = (prior_cov - v.T @ v) * y_train_std**2
            std = np.sqrt(np.maximum(np.diag(cov), kernel.noise_var)) if return_std else None
            return mean, std, cov

        # return_std only - avoid ever forming an (m,m) matrix. `kernel(t[:1], t[:1])` is a single
        # noise-free self-covariance value; constant across all m points because this kernel is
        # stationary (`is_stationary()` - a non-stationary kernel would need `np.diag(kernel(t,t))`
        # here instead, at `O(m^2)`).
        prior_var = float(kernel(t[:1], t[:1])[0, 0])
        var = (prior_var - np.sum(v**2, axis=0)) * y_train_std**2
        std = np.sqrt(np.maximum(var, kernel.noise_var))
        return mean, std, None

    def derivative(
        self, t: Array, return_std: bool = True, return_cov: bool = False,
    ) -> "tuple[np.ndarray, np.ndarray | None, np.ndarray | None]":
        """
        Mean and, on request, standard deviation and/or covariance of the DERIVATIVE process
        `f'(t)` at the query points `t` - same `return_std`/`return_cov` contract as `__call__`,
        computed from the SAME fitted `alpha_`/`L_`/kernel hyperparameters (no retraining): for a
        differentiable kernel, `f'` is itself a Gaussian process, jointly Gaussian with `f` (Solak
        et al., 2003).

            mean_deriv(t)   = gradient_X(t, X_train) @ alpha_
            cov_deriv(t,t') = hessian_XY(t,t) - v^T v,     v = L_^-1 gradient_X(t,X_train)^T

        `gradient_X`/`hessian_XY` are looked up on `self.gpr.kernel_` (the FITTED clone, optimized
        hyperparameters) - this is why they live on the kernel object rather than being hardcoded
        here: a different kernel class supplies its own, and this method never needs to know which
        kernel it's dealing with. Unlike `__call__`, there is no noise-vs-noise-free branch to
        worry about here: the derivative process never has an observation-noise term of its own
        (we never observe derivatives directly - see `RBFWithNoise.gradient_X`'s own docstring), so
        `hessian_XY`/`gradient_X` are already the right (noise-free) quantities as they stand.

        Parameters
        ------------
        t : Array, shape (m,)
            Time points at which to compute the derivative of the GP.
        return_std, return_cov : bool, optional
            Same meaning and same cost trade-off as `__call__`'s own (see its docstring).

        Returns
        -------
        mean : np.ndarray, shape (m,)
        std : np.ndarray, shape (m,) or None
            Floored at `noise_var / length_scale^2` - the derivative-scale analogue of `__call__`'s
            own `noise_var` floor (see `_FittedGP.var_deriv` in the from-scratch implementation):
            the derivative process's posterior variance can legitimately approach 0 near a densely-
            observed point, but should never be reported as MORE certain than the noise floor
            implies.
        cov : np.ndarray, shape (m, m) or None
            NOT re-floored on the diagonal (that would break its positive-semi-definiteness) - only
            `std` applies the floor, same convention as `__call__`.

        Time complexity
        ----------------
        Same structure as `__call__`: `mean` alone is `O(m * n_obs)` (no solve); `return_std` adds
        the `O(m * n_obs^2)` solve but reads the (stationary, so constant) noise-free prior
        derivative-variance in `O(1)` instead of forming an `(m,m)` matrix; `return_cov` costs the
        full `O(m^2 * n_obs)` (`v.T @ v`) plus `O(m^2)` memory.
        """
        t_flat = _as_numpy(t).ravel()
        kernel = self.gpr.kernel_
        if not hasattr(kernel, "gradient_X") or not hasattr(kernel, "hessian_XY"):
            raise NotImplementedError(
                f"kernel {kernel!r} does not implement gradient_X/hessian_XY - derivative "
                f"queries are only supported for kernels that provide their own closed-form "
                f"derivative cross-covariance (e.g. RBFWithNoise)."
            )

        X_train = self.gpr.X_train_.ravel()
        alpha = self.gpr.alpha_
        y_train_std = getattr(self.gpr, "_y_train_std", 1.0)
        noise_floor = kernel.noise_var / kernel.length_scale**2

        k_deriv_star = kernel.gradient_X(t_flat, X_train)  # (m, n_obs)
        mean = (k_deriv_star @ alpha) * y_train_std

        if not return_std and not return_cov:
            return mean, None, None

        v = solve_triangular(self.gpr.L_, k_deriv_star.T, lower=True)  # (n_obs, m)

        if return_cov:
            prior_cov = kernel.hessian_XY(t_flat, t_flat)  # (m, m)
            cov = (prior_cov - v.T @ v) * y_train_std**2
            std = np.sqrt(np.maximum(np.diag(cov), noise_floor)) if return_std else None
            return mean, std, cov

        prior_var = float(kernel.hessian_XY(t_flat[:1], t_flat[:1])[0, 0])  # stationary -> O(1)
        var = (prior_var - np.sum(v**2, axis=0)) * y_train_std**2
        std = np.sqrt(np.maximum(var, noise_floor))
        return mean, std, None
