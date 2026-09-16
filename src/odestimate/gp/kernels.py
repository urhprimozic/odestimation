"""
Custom scikit-learn-compatible kernel: RBF (squared-exponential) covariance PLUS additive
observation noise, bundled into ONE `sklearn.gaussian_process.kernels.Kernel` with three
hyperparameters - `length_scale` (l), `signal_var` (sigma_f^2), `noise_var` (sigma_n^2). This
replaces the from-scratch `_rbf_kernel` + "add noise_var to the diagonal by hand" pair used in the
original `gp.py` - here both live in ONE kernel object, fit jointly by
`GaussianProcessRegressor`'s own optimizer.

    k(t, t') = signal_var * exp(-(t-t')^2 / (2*l^2)) + noise_var * [t == t']

Deliberately NOT split into sklearn's own `ConstantKernel() * RBF() + WhiteKernel()` composition:
keeping all three hyperparameters on one object matches this project's own naming
(`length_scale`/`signal_var`/`noise_var`, same names `gp.py`'s `_FittedGP`-equivalent uses) and
avoids reading fitted values back out through a composite kernel's nested `.k1.k1`/`.k2` structure.

Gradients (`__call__(eval_gradient=True)`) are returned w.r.t. LOG-hyperparameters, not the raw
values - this is scikit-learn's own convention (`Kernel.theta` always works in log-space; see its
docstring in `sklearn/gaussian_process/kernels.py`), and it is what lets
`GaussianProcessRegressor`'s default optimizer (L-BFGS-B) use an exact gradient instead of having
to finite-difference the marginal likelihood itself - the entire reason to prefer this over the
from-scratch Nelder-Mead fit (see the chat this followed). With `theta_1=log(l)`,
`theta_2=log(signal_var)`, `theta_3=log(noise_var)`, the chain rule `d/d(log p) = p * d/dp` gives:

    dK/d(log l)          = K_rbf * (d^2 / l^2)        (d = pairwise distance)
    dK/d(log signal_var) = K_rbf
    dK/d(log noise_var)  = noise_var * I
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel


class RBFWithNoise(Kernel):
    def __init__(
        self,
        length_scale: float = 1.0,
        length_scale_bounds: "tuple[float, float]" = (1e-5, 1e5),
        signal_var: float = 1.0,
        signal_var_bounds: "tuple[float, float]" = (1e-5, 1e5),
        noise_var: float = 1e-3,
        noise_var_bounds: "tuple[float, float]" = (1e-10, 1e2),
    ):
        """
        RBF-plus-noise kernel: `k(t,t') = signal_var * exp(-(t-t')^2/(2*l^2)) + noise_var*[t==t']`.

        Parameters
        ----------
        length_scale : float
            Initial lengthscale `l` - how far apart (in time) two points must be before the prior
            considers them nearly uncorrelated.
        length_scale_bounds : (float, float)
            Bounds `GaussianProcessRegressor`'s optimizer may search `l` within (`"fixed"` also
            accepted, per `sklearn.gaussian_process.kernels.Hyperparameter`, to hold `l` constant).
        signal_var : float
            Initial signal variance `sigma_f^2` - the prior variance of the function itself
            (`k(t,t) - noise_var` for this stationary kernel).
        signal_var_bounds : (float, float)
            Bounds for `sigma_f^2`.
        noise_var : float
            Initial observation-noise variance `sigma_n^2` - added only on the diagonal.
        noise_var_bounds : (float, float)
            Bounds for `sigma_n^2`.

        Every argument is stored as a same-named plain attribute (no leading underscore, no
        renaming) - required by scikit-learn's `get_params()`/`set_params()`/`clone()` machinery,
        which inspects `__init__`'s signature and expects to find a matching attribute for each
        parameter.
        """
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.signal_var = signal_var
        self.signal_var_bounds = signal_var_bounds
        self.noise_var = noise_var
        self.noise_var_bounds = noise_var_bounds

    # Hyperparameter declarations - scikit-learn discovers these via the `hyperparameter_<name>`
    # naming convention (scanned by the base `Kernel` class), and uses them to build `theta`/
    # `bounds`/`n_dims` generically - no need to override any of those here.

    @property
    def hyperparameter_length_scale(self) -> Hyperparameter:
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_signal_var(self) -> Hyperparameter:
        return Hyperparameter("signal_var", "numeric", self.signal_var_bounds)

    @property
    def hyperparameter_noise_var(self) -> Hyperparameter:
        return Hyperparameter("noise_var", "numeric", self.noise_var_bounds)

    def __call__(self, X: np.ndarray, Y: "np.ndarray | None" = None, eval_gradient: bool = False):
        """
        Parameters
        ----------
        X : ndarray, shape (n_samples_X, n_features)
        Y : ndarray, shape (n_samples_Y, n_features), optional
            `None` (default): compute `k(X, X)` (plus the noise term on the diagonal) - the
            TRAINING covariance `GaussianProcessRegressor.fit` builds and factorizes. Given
            explicitly (e.g. at predict time, `X`=query points, `Y`=training points): compute the
            (noise-free) cross-covariance `k(X, Y)` - matches every other sklearn kernel's own
            convention that the noise term only ever belongs on the TRAINING block's diagonal, not
            on a cross block between two different point sets.
        eval_gradient : bool
            Only allowed when `Y is None` (same convention as `Y`, above: the gradient is only ever
            needed for the training kernel matrix, during hyperparameter optimization).

        Returns
        -------
        K : ndarray, shape (n_samples_X, n_samples_Y)
        K_gradient : ndarray, shape (n_samples_X, n_samples_X, 3), only if `eval_gradient=True`
            `K_gradient[..., 0]` = `dK/d(log length_scale)`, `[..., 1]` = `dK/d(log noise_var)`,
            `[..., 2]` = `dK/d(log signal_var)` - ALPHABETICAL order (`self.hyperparameters`' own
            order, not declaration order - see the in-line comment where `K_gradient` is built),
            see the module docstring for the derivation of each entry.

        Time complexity
        ----------------
        `O(n_samples_X * n_samples_Y)` for `K` itself (one `exp` per matrix entry, `cdist` is the
        same cost); `eval_gradient=True` costs another `O(n_samples_X^2)` (three more same-shape
        matrices), never `O(n^3)` - that only happens once `GaussianProcessRegressor` factorizes
        the returned `K`, outside this kernel object entirely.
        """
        X = np.atleast_2d(X)
        if Y is None:
            sq_dists = cdist(X, X, metric="sqeuclidean")
            rbf = self.signal_var * np.exp(-0.5 * sq_dists / self.length_scale**2)
            K = rbf + self.noise_var * np.eye(X.shape[0])

            if eval_gradient:
                d_length_scale = rbf * (sq_dists / self.length_scale**2)
                d_signal_var = rbf
                d_noise_var = self.noise_var * np.eye(X.shape[0])
                # `self.hyperparameters` (and hence `theta`/`bounds`, what the optimizer actually
                # indexes into) is built by scanning `hyperparameter_*` properties via `dir()`,
                # which returns them ALPHABETICALLY - "length_scale", "noise_var", "signal_var" -
                # NOT declaration order. Verified directly (`[h.name for h in
                # RBFWithNoise().hyperparameters]`) after this exact mismatch made `noise_var`'s and
                # `signal_var`'s gradient columns swap, which silently broke `GaussianProcessRegressor`'s
                # L-BFGS-B fit (it kept reporting "ABNORMAL" convergence and never left its initial
                # guess). This stacking order MUST track that alphabetical order, not the order the
                # three `d_*` locals happen to be computed in above.
                K_gradient = np.stack([d_length_scale, d_noise_var, d_signal_var], axis=-1)
                return K, K_gradient
            return K

        if eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")
        Y = np.atleast_2d(Y)
        sq_dists = cdist(X, Y, metric="sqeuclidean")
        return self.signal_var * np.exp(-0.5 * sq_dists / self.length_scale**2)

    def diag(self, X: np.ndarray) -> np.ndarray:
        """`k(t,t)` for every row of `X`: `signal_var + noise_var` everywhere (both terms are
        constant on the diagonal of this stationary kernel), shape `(n_samples,)`.

        Time complexity
        ----------------
        `O(n_samples)` - never forms the full `(n,n)` matrix just to read off its diagonal, unlike
        `np.diag(self(X))`. `GaussianProcessRegressor.predict(..., return_std=True)` calls this
        directly for exactly that reason.
        """
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self.signal_var + self.noise_var)

    def is_stationary(self) -> bool:
        return True

    # ---------------------------------------------------------------------------------------
    # Derivative-process cross-covariances (Solak et al., 2003) - NOT part of sklearn's own
    # `Kernel` interface (it has no notion of a derivative process at all), added here because
    # `GP.derivative()` needs them and they are inherently KERNEL-SPECIFIC: a different kernel
    # (Matern, periodic, ...) would need its OWN version of both methods below, generally with a
    # different closed form - there is no kernel-agnostic way to get these two quantities short of
    # automatic/numerical differentiation of `__call__` itself.

    def gradient_X(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Cross-covariance `cov(f'(X), f(Y)) = d/dX k(X,Y)`. NOT the hyperparameter gradient
        `__call__(eval_gradient=True)` returns (that one is `d/d(log theta)`, needed for fitting
        `theta` itself; this one is `d/dt`, needed to read off the DERIVATIVE process's posterior
        mean from the same `alpha_` the function-value GP already fit - no retraining, no new
        Cholesky factorization: `mean_deriv(t*) = gradient_X(t*, X_train) @ alpha_`).

        Parameters
        ----------
        X : ndarray, shape (n_X,) or (n_X, 1)
        Y : ndarray, shape (n_Y,) or (n_Y, 1)

        Returns
        -------
        ndarray, shape (n_X, n_Y)

        Time complexity
        ----------------
        `O(n_X * n_Y)`, same as `__call__` itself.
        """
        x = np.asarray(X, dtype=float).ravel()
        y = np.asarray(Y, dtype=float).ravel()
        diff = x[:, None] - y[None, :]
        k = self.signal_var * np.exp(-0.5 * (diff / self.length_scale) ** 2)
        return -k * diff / self.length_scale**2

    def hessian_XY(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Cross-covariance `cov(f'(X), f'(Y)) = d^2/(dX dY) k(X,Y)` - the derivative process's OWN
        (prior) covariance structure. At `X=Y=t*` (a single point), this is the derivative
        process's prior variance `signal_var/length_scale^2` (Rasmussen & Williams SS9.4) - matches
        plugging `diff=0` into the formula below. Combined with `gradient_X`, gives the derivative
        process's POSTERIOR covariance: `hessian_XY(t,t) - v^T v`, `v = L^-1 gradient_X(t,X_train)^T`
        (same construction `GP.derivative` uses, and the same identity `window_var` in the earlier
        from-scratch `gp.py` used for the plain function-value process).

        Parameters
        ----------
        X : ndarray, shape (n_X,) or (n_X, 1)
        Y : ndarray, shape (n_Y,) or (n_Y, 1)

        Returns
        -------
        ndarray, shape (n_X, n_Y)

        Time complexity
        ----------------
        `O(n_X * n_Y)`.
        """
        x = np.asarray(X, dtype=float).ravel()
        y = np.asarray(Y, dtype=float).ravel()
        diff = x[:, None] - y[None, :]
        k = self.signal_var * np.exp(-0.5 * (diff / self.length_scale) ** 2)
        return k * (1.0 / self.length_scale**2 - (diff / self.length_scale**2) ** 2)
