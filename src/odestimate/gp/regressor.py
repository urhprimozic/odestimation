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
    """Converts a torch tensor to numpy (detaching/moving to CPU first)"""
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a, dtype=float)


class GP1dim:
    def __init__(self, t_obs: Array, y_obs: Array, kernel: "Literal['rbf'] | Kernel" = "rbf",copy_X_train=False, **kwargs):
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
        copy_X_train : bool, optional
            Default `False`. If `True`, the training data is copied into the GP's internal memory. 
        **kwargs : dict
            Additional keyword arguments forwarded to `GaussianProcessRegressor` (e.g.
            `n_restarts_optimizer`, `normalize_y`, `alpha` for a fixed extra nugget).
        """
        if kernel == "rbf":
            kernel = RBFWithNoise()
        elif not isinstance(kernel, Kernel):
            raise NotImplementedError(
                f"kernel must be 'rbf' or a sklearn Kernel instance (not a class, another string, "
                f"or a bare callable), got {kernel!r}."
            )

        t_obs = _as_numpy(t_obs).reshape(-1, 1)
        y_obs = _as_numpy(y_obs)

        gpr = GaussianProcessRegressor(kernel=kernel, copy_X_train=copy_X_train, **kwargs)
        gpr.fit(t_obs, y_obs)
        self.gpr = gpr

    def noise_var(self) -> float:
        """Returns the fitted GP's noise variance (the additive observation noise)."""
        return self.gpr.kernel_.noise_var

    def length_scale(self) -> float:
        """Returns the fitted GP's lengthscale - together with `noise_var()`, lets a caller that
        wants a derivative-scale noise floor (`noise_var() / length_scale()**2`) build it itself,
        without reaching into `self.gpr.kernel_` directly (see `std_derivative`'s own docstring for
        why `GP1dim` no longer applies that floor internally)."""
        return self.gpr.kernel_.length_scale

    def __call__(
        self, t: Array
    ) -> Array:
        """
        Predicts the mean 

        Parameters
        ------------
        t : Array, shape (m,)
            Time points at which to predict the GP.

        Returns
        -------
        mean : np.ndarray, shape (m,)
            The predicted mean of the GP at the query points `t`.
        Time complexity
        ----------------
        `mean` alone: `O(m * n_obs)`
        """
        t = _as_numpy(t).reshape(-1, 1)
        kernel = self.gpr.kernel_
        X_train = self.gpr.X_train_
        alpha = self.gpr.alpha_
        y_train_std = getattr(self.gpr, "_y_train_std", 1.0)
        y_train_mean = getattr(self.gpr, "_y_train_mean", 0.0)

        k_star = kernel(t, X_train)  # (m, n_obs), noise-free (Y given explicitly)
        mean = k_star @ alpha * y_train_std + y_train_mean

        return mean

    def mean(self, t: Array) -> Array:
        return self.__call__(t)

    def std(self, t: Array) -> Array:
        """RAW (unfloored) posterior std of the latent `x_hat(t)`. See `__init__`'s module-level
        discussion (and the chat this followed) for why this does NOT floor at `noise_var()` -
        that's a policy decision for whichever caller needs a `1/sqrt(var)` weight, not this
        class's own concern."""
        t = _as_numpy(t).reshape(-1, 1)
        kernel = self.gpr.kernel_
        X_train = self.gpr.X_train_
        y_train_std = getattr(self.gpr, "_y_train_std", 1.0)

        k_star = kernel(t, X_train)  # (m, n_obs), noise-free (Y given explicitly)
        v = solve_triangular(self.gpr.L_, k_star.T, lower=True)  # (n_obs, m)

        prior_var = float(kernel(t[:1], t[:1])[0, 0])
        var = (prior_var - np.sum(v**2, axis=0)) * y_train_std**2
        return np.sqrt(var)

    def cov(self, t: Array) -> Array:
        """RAW (unfloored) posterior covariance of the latent `x_hat(t)` - see `std`'s own
        docstring for why no flooring happens here either."""
        t = _as_numpy(t).reshape(-1, 1)
        kernel = self.gpr.kernel_
        X_train = self.gpr.X_train_
        y_train_std = getattr(self.gpr, "_y_train_std", 1.0)

        k_star = kernel(t, X_train)  # (m, n_obs), noise-free (Y given explicitly)
        v = solve_triangular(self.gpr.L_, k_star.T, lower=True)  # (n_obs, m)

        prior_cov = kernel(t, t)  # noise-free, (m, m) - see docstring
        return (prior_cov - v.T @ v) * y_train_std**2

    def _check_derivative_support(self) -> Kernel:
        kernel = self.gpr.kernel_
        if not hasattr(kernel, "gradient_X") or not hasattr(kernel, "hessian_XY"):
            raise NotImplementedError(
                f"kernel {kernel!r} does not implement gradient_X/hessian_XY - derivative "
                f"queries are only supported for kernels that provide their own closed-form "
                f"derivative cross-covariance (e.g. RBFWithNoise)."
            )
        return kernel

    def derivative(self, t: Array) -> Array:
        """
        Mean of the derivative process `x_hat'(t)` at the query points `t` - for a differentiable
        kernel, `x_hat'` is itself a Gaussian process, jointly Gaussian with `x_hat` (Solak et al.,
        2003): `mean_deriv(t) = gradient_X(t, X_train) @ alpha_`, the SAME fitted `alpha_` (no
        retraining). `gradient_X` is looked up on `self.gpr.kernel_` (the FITTED clone) rather than
        hardcoded here - a different kernel class supplies its own, and this method never needs to
        know which kernel it's dealing with.

        Parameters
        ------------
        t : Array, shape (m,)
            Time points at which to compute the derivative of the GP.

        Returns
        -------
        np.ndarray, shape (m,)

        Time complexity
        ----------------
        `O(m * n_obs)` - no triangular solve, same reasoning as `mean`.
        """
        t_flat = _as_numpy(t).ravel()
        kernel = self._check_derivative_support()
        X_train = self.gpr.X_train_.ravel()
        y_train_std = getattr(self.gpr, "_y_train_std", 1.0)

        k_deriv_star = kernel.gradient_X(t_flat, X_train)  # (m, n_obs)
        return (k_deriv_star @ self.gpr.alpha_) * y_train_std

    def std_derivative(self, t: Array) -> Array:
        """RAW (unfloored) posterior std of the derivative process `x_hat'(t)` - see `std`'s own
        docstring for why no flooring happens here (same policy: that's for the caller to decide).
        `O(m * n_obs^2)` (the triangular solve dominates); the noise-free prior derivative-variance
        is read off in `O(1)` (stationary kernel - `hessian_XY(t[:1],t[:1])`), not `O(m^2)`."""
        t_flat = _as_numpy(t).ravel()
        kernel = self._check_derivative_support()
        X_train = self.gpr.X_train_.ravel()
        y_train_std = getattr(self.gpr, "_y_train_std", 1.0)

        k_deriv_star = kernel.gradient_X(t_flat, X_train)  # (m, n_obs)
        v = solve_triangular(self.gpr.L_, k_deriv_star.T, lower=True)  # (n_obs, m)

        prior_var = float(kernel.hessian_XY(t_flat[:1], t_flat[:1])[0, 0])  # stationary -> O(1)
        var = (prior_var - np.sum(v**2, axis=0)) * y_train_std**2
        return np.sqrt(var)

    def cov_derivative(self, t: Array) -> Array:
        """RAW (unfloored) posterior covariance of the derivative process `x_hat'(t)` - see `cov`'s
        own docstring for why no flooring happens here. `O(m^2 * n_obs)` (the `v.T @ v` term) plus
        `O(m^2)` memory, same cost structure as `cov`."""
        t_flat = _as_numpy(t).ravel()
        kernel = self._check_derivative_support()
        X_train = self.gpr.X_train_.ravel()
        y_train_std = getattr(self.gpr, "_y_train_std", 1.0)

        k_deriv_star = kernel.gradient_X(t_flat, X_train)  # (m, n_obs)
        v = solve_triangular(self.gpr.L_, k_deriv_star.T, lower=True)  # (n_obs, m)

        prior_cov = kernel.hessian_XY(t_flat, t_flat)  # (m, m)
        return (prior_cov - v.T @ v) * y_train_std**2

class GP:
    """
    Multi-variable wrapper around `GP1dim`: fits ONE independent `GP1dim` per state variable, each
    with its OWN independently-optimized hyperparameters - NOT sklearn's own multi-output `y`
    support (a 2D `y` array there would fit a SINGLE shared kernel/hyperparameter set across every
    output column, solving multiple right-hand sides against the same `K`). Cross-variable
    covariance is therefore always exactly 0 by construction (no `GP1dim` ever sees another
    variable's data) - matches gradient matching's own design, where smoothing happens
    independently per variable and the ODE is the only place variables are ever allowed to
    interact (see the chat this followed, incl. why a genuine multi-output/coregionalized kernel
    was considered and rejected for this project).
    """

    def __init__(
        self, t_obs: Array, y_obs: Array, kernel: "Literal['rbf'] | Kernel" = "rbf",
        copy_X_train: bool = False, **kwargs,
    ):
        """
        Parameters
        ------------
        t_obs : Array, shape (n,)
            SHARED time grid - every variable is assumed observed at the same `n` times. If your
            variables are sampled at different times/frequencies, fit separate `GP1dim`s directly
            instead (one per variable, each with its own `t_obs`) - this wrapper doesn't support
            per-variable grids.
        y_obs : Array, shape (n_vars, n)
            Row `i` is variable `i`'s own observed series - `(n_vars, n_points)`, the SAME
            convention `state_at_collocation`/`deriv_at_collocation` use elsewhere in this project
            (not `(n_points, n_vars)`).
        kernel, copy_X_train, **kwargs :
            Forwarded to EVERY `GP1dim` - same kernel CLASS/prior family and solver options for
            every variable, but each still gets its OWN independently-fit hyperparameter VALUES
            (they are never shared or averaged across variables).
        """
        y_obs = _as_numpy(y_obs)
        self.gps = [
            GP1dim(t_obs, y_obs[i, :], kernel=kernel, copy_X_train=copy_X_train, **kwargs)
            for i in range(y_obs.shape[0])
        ]

    def __len__(self) -> int:
        return len(self.gps)

    def noise_var(self) -> np.ndarray:
        """`(n_vars,)` - each variable's own fitted noise variance."""
        return np.array([gp.noise_var() for gp in self.gps])

    def length_scale(self) -> np.ndarray:
        """`(n_vars,)` - each variable's own fitted lengthscale."""
        return np.array([gp.length_scale() for gp in self.gps])

    def __call__(self, t: Array) -> Array:
        return self.mean(t)

    def mean(self, t: Array) -> Array:
        """`(m, n_vars)` - column `i` is variable `i`'s own posterior mean at `t`."""
        return np.column_stack([gp.mean(t) for gp in self.gps])

    def std(self, t: Array) -> Array:
        """`(m, n_vars)`, RAW (unfloored) - see `GP1dim.std`'s own docstring for why."""
        return np.column_stack([gp.std(t) for gp in self.gps])

    def cov(self, t: Array) -> Array:
        """
        `(n_vars, m, m)` - `cov(t)[i]` is variable `i`'s own `(m,m)` posterior covariance. NOT a
        joint `(m*n_vars, m*n_vars)` matrix: since cross-variable covariance is always exactly 0
        (see the class docstring), a block-diagonal joint matrix would be almost entirely wasted
        zeros - this stacks only the non-trivial per-variable diagonal blocks.
        """
        return np.stack([gp.cov(t) for gp in self.gps])

    def derivative(self, t: Array) -> Array:
        """`(m, n_vars)` - column `i` is variable `i`'s own derivative-process mean at `t`."""
        return np.column_stack([gp.derivative(t) for gp in self.gps])

    def std_derivative(self, t: Array) -> Array:
        """`(m, n_vars)`, RAW (unfloored) - see `GP1dim.std_derivative`'s own docstring."""
        return np.column_stack([gp.std_derivative(t) for gp in self.gps])

    def cov_derivative(self, t: Array) -> Array:
        """`(n_vars, m, m)` - see `cov`'s own docstring for why not one joint block matrix."""
        return np.stack([gp.cov_derivative(t) for gp in self.gps])
    #TODO ADD OTHERS