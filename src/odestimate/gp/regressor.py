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
            # pop (not get) - kernel_kwargs must not leak into the **kwargs forwarded to
            # GaussianProcessRegressor below, which has no such parameter
            kernel_kwargs = kwargs.pop("kernel_kwargs", None)
            if kernel_kwargs is not None:
                kernel = RBFWithNoise(**kernel_kwargs)
            else:
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

    def cov_state_derivative(self, t: Array) -> Array:
        """
        `Cov[X(t), X'(t)]` at the SAME `t` (paired, not a matrix) - the cross term between the
        latent function and its own derivative, needed to propagate the joint state/derivative
        uncertainty through a nonlinear `f` via the delta method (see
        `notes/gradient-matching-pruning.md` Trditev 3.1 in the PyBM repo for the full derivation).

        Zero in the PRIOR for any stationary kernel (the derivative of an even function at 0 is 0)
        - NOT necessarily zero in the posterior, since `X(t)` and `X'(t)` are both conditioned on
        the same training data and so can end up correlated through it:

            cov(t) = -v(t)^T v'(t)

        where `v(t)` (from `std`) and `v'(t)` (from `std_derivative`) are the same two triangular-
        solve projections those methods already compute - this reuses that work, not new machinery.

        Parameters
        ------------
        t : Array, shape (m,)

        Returns
        -------
        np.ndarray, shape (m,)

        Time complexity
        ----------------
        `O(m * n_obs^2)` - one triangular solve for `v`, one for `v'` (same cost as `std`/
        `std_derivative` individually), plus a cheap `O(m * n_obs)` dot product.
        """
        t = _as_numpy(t).reshape(-1, 1)
        t_flat = t.ravel()
        kernel = self._check_derivative_support()
        X_train = self.gpr.X_train_
        X_train_flat = X_train.ravel()
        y_train_std = getattr(self.gpr, "_y_train_std", 1.0)

        k_star = kernel(t, X_train)  # (m, n_obs), noise-free
        k_deriv_star = kernel.gradient_X(t_flat, X_train_flat)  # (m, n_obs)
        v = solve_triangular(self.gpr.L_, k_star.T, lower=True)  # (n_obs, m)
        v_deriv = solve_triangular(self.gpr.L_, k_deriv_star.T, lower=True)  # (n_obs, m)

        # Prior cross-covariance is exactly 0 at the same t (stationary + even kernel) - only the
        # posterior projection term survives.
        return -np.sum(v * v_deriv, axis=0) * y_train_std**2

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

    @classmethod
    def _from_gps(cls, gps: "list[GP1dim]") -> "GP":
        """Internal constructor bypassing `__init__`'s shared-`t_obs` assumption: builds a `GP`
        directly from already-fitted `GP1dim` instances, one per variable, each possibly trained
        on its OWN distinct `t_obs` subset. `__init__` itself intentionally forbids this (see its
        own docstring - every other caller in this project relies on all variables sharing one time
        grid); used by `heuristic_gp_fit(shared_points=False)`, where that no longer holds."""
        self = cls.__new__(cls)
        self.gps = list(gps)
        return self

    def noise_var(self) -> np.ndarray:
        """`(n_vars,)` - each variable's own fitted noise variance."""
        return np.array([gp.noise_var() for gp in self.gps])

    def length_scale(self) -> np.ndarray:
        """`(n_vars,)` - each variable's own fitted lengthscale."""
        return np.array([gp.length_scale() for gp in self.gps])

    def __call__(self, t: Array) -> Array:
        return self.mean(t)

    def mean(self, t: Array) -> Array:
        """`(n_timepoints, n_vars)` - column `i` is variable `i`'s own posterior mean at `t`."""
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

    def cov_state_derivative(self, t: Array) -> Array:
        """`(m, n_vars)` - column `i` is variable `i`'s own `Cov[X(t), X'(t)]` (paired, per point) -
        see `GP1dim.cov_state_derivative`'s own docstring. Cross-VARIABLE terms are not part of
        this (always exactly 0 - independent `GP1dim`s, see the class docstring)."""
        return np.column_stack([gp.cov_state_derivative(t) for gp in self.gps])


def _select_diverse_worst(t_pool: np.ndarray, errors: np.ndarray, n_select: int, eps: float) -> np.ndarray:
    """
    Indices (into `t_pool`/`errors`) of up to `n_select` worst (highest-`errors`) points, picked
    GREEDILY in descending-error order - skip a candidate if it's closer than `eps` to a point
    ALREADY PICKED IN THIS SAME CALL (not to any pre-existing training point). Without this, a
    single bad neighborhood could claim an entire round's batch instead of spreading coverage.
    `eps<=0`: no spacing constraint, plain top-`n_select` by error. May return FEWER than
    `n_select` indices if `eps` is too large relative to how many well-spaced candidates exist in
    `t_pool` - never raises, the caller decides what to do with a short result.
    """
    order = np.argsort(-errors)
    selected: "list[int]" = []
    for idx in order:
        if len(selected) >= n_select:
            break
        if eps <= 0 or all(abs(t_pool[idx] - t_pool[s]) >= eps for s in selected):
            selected.append(int(idx))
    return np.array(selected, dtype=int)


def _heuristic_gp_fit_core(
    t_obs: np.ndarray, y_obs: np.ndarray, max_points: int, n_init: int, n_added: int,
    eps: float, rng: np.random.Generator, pbar=None, **gp_kwargs,
) -> "tuple[GP, np.ndarray]":
    """
    Core greedy loop shared by both `heuristic_gp_fit` modes - see its own docstring for the
    algorithm. `y_obs` may carry one or several variables' rows (shape `(n_vars_here, n)`); error
    is always summed across whichever rows are present, so calling this with a SINGLE row (shape
    `(1, n)`) - as `shared_points=False` does, once per variable - reduces to that one variable's
    own error with no separate code path needed. `pbar`, if given, is updated in place but NOT
    created or closed here - the caller owns its lifecycle (it may want a fresh bar per variable).
    `**gp_kwargs` is forwarded to every `GP(...)` refit unchanged (e.g. `kernel`, `kernel_kwargs`,
    `n_restarts_optimizer`) - same kernel/solver settings on every round, only the training set grows.

    Returns `(gp, train_mask)` - `train_mask` (boolean, shape `(n,)`, indexing into the `t_obs`/
    `y_obs` this was called with) lets the caller recover the exact training points `gp` was fit
    on, e.g. `t_obs[train_mask]`/`y_obs[:, train_mask]`.
    """
    n_total = len(t_obs)
    all_idx = np.arange(n_total)
    train_idx = rng.choice(all_idx, size=min(n_init, max_points), replace=False)
    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[train_idx] = True

    gp = GP(t_obs[train_mask], y_obs[:, train_mask], **gp_kwargs)

    while train_mask.sum() < max_points:
        pool_idx = all_idx[~train_mask]
        t_pool = t_obs[pool_idx]
        y_pool = y_obs[:, pool_idx]

        pred = gp.mean(t_pool)  # (m, n_vars_here)
        errors = np.sum((pred - y_pool.T) ** 2, axis=1)  # (m,), summed across whatever rows y_obs has

        n_to_add = min(n_added, max_points - int(train_mask.sum()))
        chosen = _select_diverse_worst(t_pool, errors, n_to_add, eps)
        if len(chosen) == 0:
            break  # eps too large relative to remaining pool spacing - can't add any more safely

        train_mask[pool_idx[chosen]] = True
        gp = GP(t_obs[train_mask], y_obs[:, train_mask], **gp_kwargs)
        if pbar is not None:
            pbar.update(len(chosen))

    return gp, train_mask


def heuristic_gp_fit(
    t_obs: Array, y_obs: Array, max_points: int, n_init: int, n_added: int,
    eps: float = 0.0, verbose: int = 0, shared_points: bool = True, **gp_kwargs,
) -> "tuple[GP, np.ndarray, np.ndarray] | tuple[GP, list[np.ndarray], list[np.ndarray]]":
    """
    Greedy, ERROR-driven training-point selection, instead of a random or evenly-spaced subsample.

    NOT active learning in the classical sense: there's no unobserved "true" value to guess at -
    every candidate point's own `y` is already known (this only decides which of the ALREADY-
    OBSERVED points to bother feeding into the (cheaper, sparser) GP fit). Closer to adaptive mesh
    refinement: repeatedly add the points where the CURRENT fit is most WRONG (measured against
    the actually-observed `y` there), not where the model merely reports itself as uncertain. This
    sidesteps a real failure mode (see the chat this followed, and `notes/gradient-matching-
    pruning.md`'s own SS6 in the PyBM repo): with sparse training data, a GP's own posterior std
    can stay small EXACTLY where its mean is badly wrong (checked directly - a plain random 25-50
    point subsample of Bled's `phyto.conc` gave z-scores >20 at its worst-missed point, i.e. the
    reported uncertainty understated the true error by over an order of magnitude). Picking new
    training points by REAL error on a held-out pool, rather than by the model's own (possibly
    miscalibrated) self-reported uncertainty, avoids trusting exactly the quantity that was just
    shown to be unreliable when data is sparse.

    Algorithm
    ---------
    1. `n_init` points chosen uniformly AT RANDOM from `(t_obs, y_obs)` seed the training set.
    2. While `len(train) < max_points`:
       a. Evaluate the CURRENT fit's error at every point still in the pool (`t_obs` minus
          `train`) - summed squared error across every state variable at that time point.
       b. Greedily pick up to `n_added` of the worst (highest-error) pool points via
          `_select_diverse_worst` (spacing them at least `eps` apart from EACH OTHER within this
          same round - see its own docstring). Move them from pool to train.
       c. Re-fit a GP on the now-larger training set - a fresh hyperparameter search each round,
          no warm start.
    Returns the GP from the LAST such fit (no separate final re-fit needed).

    Parameters
    ----------
    t_obs : Array, shape (n,)
    y_obs : Array, shape (n,) or (n_vars, n)
        Single- or multi-variable observed data - 1D input is treated as one variable (reshaped to
        `(1, n)`), matching `GP`'s own `(n_vars, n)` convention either way. Always returns a `GP`
        (even for a single variable), for a consistent return type.
    max_points : int
        Target training-set size once the loop finishes - capped at `len(t_obs)` if larger (can't
        train on more points than were actually observed).
    n_init : int
        Random seed points before the greedy loop starts (capped at `max_points`).
    n_added : int
        How many points to add per round, before re-fitting - trades off refit COUNT (fewer,
        cheaper rounds for a larger `n_added`) against how "fresh" each pick in a large batch is
        (an entire batch is chosen against the SAME fit, so later picks within one big batch are
        less well-motivated than picks made one at a time against a just-updated fit).
    eps : float, optional
        Minimum spacing required BETWEEN points added in the SAME round (see
        `_select_diverse_worst`). Default `0` - no spacing constraint.
    verbose : int, optional
        `>0` shows a `tqdm` progress bar (points added so far / `max_points`) - one bar for the
        whole run if `shared_points=True`, one bar per variable otherwise.
    shared_points : bool, optional
        Default `True`: every variable is trained on the SAME growing point set, picked by their
        SUMMED per-point error (a point can be added because it's bad for any one variable, even if
        the others already fit it fine there) - one shared `train_mask`, `n_vars` GP1dims that all
        happen to share their `t_obs`. `False`: each variable instead runs its OWN independent copy
        of the algorithm end-to-end, picking points by its OWN error alone (never the other
        variables') - `n_vars` genuinely different training subsets, one per variable. Costs `n_vars`
        times the GP refits of the shared-points path, in exchange for not forcing e.g. a
        fast-changing variable's own worst points onto a slowly-varying variable's fit (or vice
        versa).
    **gp_kwargs :
        Forwarded to every `GP(...)` refit unchanged (e.g. `kernel`, `kernel_kwargs`,
        `n_restarts_optimizer`) - same as passing them directly to `GP.__init__`.

    Returns
    -------
    gp : GP
        Fit on the final training set(s).
    t_train, y_train :
        The final training points `gp` was fit on. `shared_points=True`: single arrays, shapes
        `(n_train,)` and `(n_vars, n_train)` - a subset of `t_obs`/`y_obs`, same convention as the
        inputs. `shared_points=False`: lists of length `n_vars` (one `(n_train_i,)`/`(n_train_i,)`
        pair per variable - sizes can differ across variables, since each ran its own independent
        loop and may have stopped early for a different reason, e.g. `eps`).
    """
    t_obs = _as_numpy(t_obs)
    y_obs = np.atleast_2d(_as_numpy(y_obs))  # (n_vars, n)
    max_points = min(max_points, len(t_obs))
    rng = np.random.default_rng()

    if shared_points:
        pbar = None
        if verbose > 0:
            from tqdm import tqdm
            pbar = tqdm(total=max_points, initial=min(n_init, max_points), desc="heuristic_gp_fit")
        gp, train_mask = _heuristic_gp_fit_core(
            t_obs, y_obs, max_points, n_init, n_added, eps, rng, pbar, **gp_kwargs
        )
        if pbar is not None:
            pbar.close()
        return gp, t_obs[train_mask], y_obs[:, train_mask]

    # shared_points=False: each variable picks its own training subset, independently of every
    # other variable's own error - one full run of the same core loop per row of y_obs.
    gps = []
    t_trains = []
    y_trains = []
    for i in range(y_obs.shape[0]):
        pbar = None
        if verbose > 0:
            from tqdm import tqdm
            pbar = tqdm(
                total=max_points, initial=min(n_init, max_points),
                desc=f"heuristic_gp_fit (var {i})",
            )
        gp_i, train_mask = _heuristic_gp_fit_core(
            t_obs, y_obs[i : i + 1, :], max_points, n_init, n_added, eps, rng, pbar, **gp_kwargs
        )
        if pbar is not None:
            pbar.close()
        gps.append(gp_i.gps[0])
        t_trains.append(t_obs[train_mask])
        y_trains.append(y_obs[i, train_mask])

    return GP._from_gps(gps), t_trains, y_trains