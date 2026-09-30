# gaussian processes + gradient matching
# interpolate the data using a GP -> interpolant x(t)
# compare x'(t) with f(x(t), t, theta) to estimate theta
from typing import Literal

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.stats import t as student_t
from torch.func import jacrev, vjp

from odestimate.gp.regressor import GP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64

type Engine = Literal["torch", "scipy"]


def _as_tensor(a) -> torch.Tensor:
    return torch.as_tensor(a, dtype=DTYPE, device=DEVICE)


def gradient_matching(ts, xs, dxs, f, theta_0, engine: Engine = "torch", **kwargs):
    """
    Gradient matching on PRECOMPUTED state and derivative values: finds theta minimizing

        sum_i || dxs_i - f(xs_i, ts_i, theta) ||^2

    Parameters
    ----------
    ts : array-like, shape (n,)
        Time points.
    xs : array-like, shape (n, d)
        x(t_i), e.g. a GP mean (see `precompute`).
    dxs : array-like, shape (n, d)
        x'(t_i), e.g. a GP mean derivative (see `precompute`).
    f : callable
        f(x, t, theta) -> dx/dt, shape (n, d). Torch tensors in, torch out for `engine="torch"`;
        numpy in, numpy out for `engine="scipy"`.
    theta_0 : array-like, shape (p,)
        Initial guess for theta.
    engine : "torch" or "scipy"
        "torch": exact Jacobian w.r.t. theta by reverse-mode autodiff.
        "scipy": pure numpy, no torch involved; Jacobian by scipy's own finite
        differences (`least_squares`' default `jac="2-point"`, override via `jac=`).
    **kwargs :
        Forwarded to `scipy.optimize.least_squares` (e.g. `bounds`).

    Returns
    -------
    scipy.optimize.OptimizeResult
        `.x` holds the estimated theta.
    """
    if engine == "scipy":
        ts, xs, dxs = np.asarray(ts, float), np.asarray(xs, float), np.asarray(dxs, float)
        return least_squares(lambda theta: (dxs - f(xs, ts, theta)).reshape(-1), x0=theta_0, **kwargs)

    ts, xs, dxs = _as_tensor(ts), _as_tensor(xs), _as_tensor(dxs)

    def residual(theta: torch.Tensor) -> torch.Tensor:
        return (dxs - f(xs, ts, theta)).reshape(-1)

    # least_squares itself is scipy, so theta comes in and residuals go out as numpy; on CPU both
    # conversions are zero-copy views.
    def fun(theta_np: np.ndarray) -> np.ndarray:
        return residual(_as_tensor(theta_np)).detach().cpu().numpy()

    # reverse mode even when p < n*d: jacrev runs f's Python once and vmaps only the backward pass
    # over the recorded graph, while jacfwd runs f under vmap - measured 3x slower on PyBM models.
    def jac(theta_np: np.ndarray) -> np.ndarray:
        return jacrev(residual)(_as_tensor(theta_np)).detach().cpu().numpy()

    return least_squares(fun, x0=theta_0, jac=jac, **kwargs)


def gradient_matching_gp(t_obs, x_obs, f, theta_0, gp=None, **kwargs):
    """
    Gradient matching for parameter estimation in ODEs using Gaussian Processes.
    Finds the theta that best explains the ODE system

        x'(t) = f(x(t), t, theta)

    by matching the GP's own derivative against f, instead of actually integrating the ODE.

    Parameters
    ----------
    t_obs : array-like, shape (n_samples,)
        Time points at which the residual is built. Also used to FIT the GP, unless `gp` is
        given - in that case these can be ANY query points (e.g. `uniform_trust_region`'s randomly
        sampled ones), decoupled from whatever data the GP was actually trained on.
    x_obs : array-like, shape (n_features, n_samples), or None
        The observed state values, one row per state variable (same convention as
        `odestimate.gp.regressor.GP`). Only used to fit a fresh GP - ignored (may be `None`) if
        `gp` is given.
    f : callable
        The ODE right-hand side, f(x, t, theta) -> dx/dt. Must be written in torch (it gets
        differentiated w.r.t. theta).
    theta_0 : array-like
        Initial guess for theta.
    gp : odestimate.gp.regressor.GP, optional
        A GP already fit elsewhere - reuse it instead of fitting a new one on `x_obs`.
    **kwargs : dict
        Forwarded to `gradient_matching` (and from there to `scipy.optimize.least_squares`).

    Returns
    -------
    scipy.optimize.OptimizeResult
        `.x` holds the estimated theta.
    """
    if gp is None:
        gp = GP(t_obs, x_obs)
    return gradient_matching(t_obs, gp.mean(t_obs), gp.derivative(t_obs), f, theta_0, **kwargs)


def precompute(ts, gp, engine: Engine = "torch"):
    """
    Everything `gradient_matching` and `confidence_interval` need from the GP at `ts`. None of it
    depends on theta or on the candidate `f`, so compute it ONCE per (GP, test points) and share it
    across every candidate - the per-candidate loop then never touches the GP.

    Parameters
    ----------
    ts : array-like, shape (n,)
    gp : odestimate.gp.regressor.GP
    engine : "torch" or "scipy"
        Return torch tensors or numpy arrays - whichever the per-candidate loop will use.

    Returns
    -------
    xs, dxs, x_vars, dx_vars, covs : each shape (n, d)
        GP posterior mean of x(t) and x'(t), and the raw (unfloored) posterior
        Var[x(t)], Var[x'(t)], Cov[x(t), x'(t)] at every t in `ts`.

    Time complexity
    ----------------
    `O(n * n_gp^2)` - triangular solves against the GP's Cholesky factor.
    """
    values = (
        gp.mean(ts),
        gp.derivative(ts),
        gp.std(ts) ** 2,
        gp.std_derivative(ts) ** 2,
        gp.cov_state_derivative(ts),
    )
    convert = _as_tensor if engine == "torch" else (lambda a: np.asarray(a, dtype=float))
    return tuple(convert(v) for v in values)


def _fd_Fx_transpose_r(f, xs, ts, theta, fx, r):
    """u_i = F_x(t_i)^T r_i by forward differences: one extra (batched) f evaluation per state
    dimension, since f acts pointwise and perturbing column k of every row at once perturbs each
    point independently."""
    u = np.empty_like(r)
    for k in range(xs.shape[1]):
        h = np.sqrt(np.finfo(float).eps) * (1.0 + np.abs(xs[:, k]))
        xp = xs.copy()
        xp[:, k] += h
        dF_dxk = (f(xp, ts, theta) - fx) / h[:, None]  # (n, d): row i is dF(x_i)/dx_ik
        u[:, k] = (dF_dxk * r).sum(axis=1)
    return u


def confidence_interval(ts, xs, dxs, f, theta, x_vars, dx_vars, covs, p_value=0.01, engine: Engine = "torch"):
    """
    Confidence interval of the gradient matching L2 loss

        L_n(theta) +- t_{n-1}(1 - p_value / 2) * SE,    SE = sqrt(Var_A + Var_B)

    Var_A - variance due to subsampling (sample variance of rho_i / n).
    Var_B - variance due to the GP's uncertainty in x and x' (delta method):
            (1/n^2) sum_i 4 r_i^T Cov[delta r_i] r_i,
            Cov[delta r_i] = S_d - F_x S_c - S_c F_x^T + F_x S F_x^T  (S's diagonal).
    See `notes/gradient-matching-pruning.md` (PyBM repo).

    Parameters
    ----------
    ts : array-like, shape (n,)
        Test time points.
    xs, dxs : array-like, shape (n, d)
        x(t_i) and x'(t_i) (see `precompute`).
    f : callable
        f(x, t, theta) -> dx/dt, shape (n, d) - torch for `engine="torch"`, numpy for "scipy".
        Must act pointwise (row i of the output depends only on row i of x).
    theta : array-like, shape (p,)
        Theta at which to compute the interval.
    x_vars : array-like, shape (n, d)
        sigma^2 = Var(x(t) | data) at `ts`.
    dx_vars : array-like, shape (n, d)
        sigma_d^2 = Var(x'(t) | data) at `ts`.
    covs : array-like, shape (n, d)
        c = Cov(x(t), x'(t) | data) at `ts`.
    p_value : float
        Significance level alpha of the (1 - alpha) interval.
    engine : "torch" or "scipy"
        How u_i = F_x(t_i)^T r_i is obtained: "torch" - one vjp; "scipy" - forward differences
        (d extra evaluations of f).

    Time complexity
    -----------------
    `O(n * d)` arithmetic plus one forward and one reverse pass of `f` over the batch ("torch"),
    or d + 1 forward passes ("scipy"). F_x is never built: since f acts pointwise, its Jacobian
    over the batch is block-diagonal, so a single vjp with cotangent r yields u_i for every i.

    Returns
    ---------
    (lower, upper) : tuple[float, float]
    """
    if engine == "scipy":
        ts, xs, dxs = np.asarray(ts, float), np.asarray(xs, float), np.asarray(dxs, float)
        theta = np.asarray(theta, float)
        fx = f(xs, ts, theta)
        r = dxs - fx
        u = _fd_Fx_transpose_r(f, xs, ts, theta, fx, r)
    else:
        ts, xs, dxs, theta = _as_tensor(ts), _as_tensor(xs), _as_tensor(dxs), _as_tensor(theta)
        fx, f_vjp = vjp(lambda x: f(x, ts, theta), xs)
        r_t = dxs - fx
        (u_t,) = f_vjp(r_t)
        r, u = r_t.detach().cpu().numpy(), u_t.detach().cpu().numpy()

    x_vars, dx_vars, covs = (a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a, float) for a in (x_vars, dx_vars, covs))
    n = len(r)

    rho = (r**2).sum(axis=1)  # (n,)
    L_n = float(rho.mean())
    var_A = float(rho.var(ddof=1)) / n if n > 1 else 0.0

    # r^T Cov[delta r] r, expanded for diagonal S, S_d, S_c; clamped at 0 per point (PSD in exact
    # arithmetic, floating point can dip a hair negative).
    quad = (dx_vars * r**2 - 2 * covs * r * u + x_vars * u**2).sum(axis=1)
    var_B = float(np.clip(4.0 * quad, 0.0, None).sum()) / n**2

    se = float(np.sqrt(var_A + var_B))
    t_star = float(student_t.ppf(1 - p_value / 2, df=max(n - 1, 1)))
    return L_n - t_star * se, L_n + t_star * se


def confidence_interval_gp(gp, f, t_samples, theta, p_value):
    """`confidence_interval` straight from a GP - convenience wrapper around `precompute`."""
    xs, dxs, x_vars, dx_vars, covs = precompute(t_samples, gp)
    return confidence_interval(t_samples, xs, dxs, f, theta, x_vars, dx_vars, covs, p_value)


def uniform_trust_region(t_obs, x_obs, f, theta_0, n_samples, p_value=0.1, gp=None, **kwargs):
    """
    Fast gradient matching for screening/pruning candidate `f`'s: fits the GP on the FULL data
    (best possible interpolant), then runs gradient matching on only `n_samples` time points
    sampled UNIFORMLY AT RANDOM from the observed range - plus a confidence interval for how far
    the resulting loss can plausibly be from the "full" (continuum, true-trajectory) loss. See
    `notes/gradient-matching-pruning.md` (PyBM repo) for the full derivation; not meant as a final
    fit, only to rank/eliminate candidates quickly before spending real effort on the survivors.

    Parameters
    ----------
    t_obs, x_obs : as in `gradient_matching_gp` - the FULL observed data. The GP is fit on all of it.
    f : as in `gradient_matching`.
    theta_0 : as in `gradient_matching`.
    n_samples : int
        Number of test points, drawn i.i.d. uniformly from `[min(t_obs), max(t_obs)]` - genuinely
        continuous query points into the already-fitted GP, NOT a subset of `t_obs`'s own indices.
    p_value : float, optional
        Significance level alpha for the resulting `(1 - alpha)` confidence interval. Default
        `0.1` (90% CI).
    gp : odestimate.gp.regressor.GP, optional
        A GP already fit elsewhere - reuse it instead of fitting a new one on `x_obs`.
    **kwargs : dict
        Forwarded to `gradient_matching` (and from there to `scipy.optimize.least_squares`).

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Same as `gradient_matching`'s own return value, fit on the `n_samples` sampled points.
    interval : tuple[float, float]
        `(lower, upper)` confidence bound on the L2 loss achievable with the full data and the
        true (not interpolated) trajectory - see `confidence_interval`.
    """
    t_obs = np.asarray(t_obs, dtype=float)
    t_samples = np.random.default_rng().uniform(t_obs.min(), t_obs.max(), size=n_samples)
    if gp is None:
        gp = GP(t_obs, x_obs)

    xs, dxs, x_vars, dx_vars, covs = precompute(t_samples, gp)
    result = gradient_matching(t_samples, xs, dxs, f, theta_0, **kwargs)
    interval = confidence_interval(t_samples, xs, dxs, f, result.x, x_vars, dx_vars, covs, p_value)
    return result, interval
