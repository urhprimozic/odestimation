import functools
from typing import Iterable, Literal
import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import NonlinearConstraint

def _torch(f, ts, xs, n_subintervals,theha_0, unobserved,unobserved_inits,int_threshold, **kwargs):
    pass

def _scipy_trajectory(f, t_0, t_1, t_loss, x0, theta, int_threshold, **kwargs):
    """
    Solves x' = f(x, t, θ) with x(t_0) = x0 on [t_0, t_1], for fixed θ.

    If |x| reaches int_threshold, integration stops there and the state it reached (of size
    int_threshold) is held to t_1: a diverging trajectory then costs a large, finite loss instead of
    overflowing or grinding on with ever smaller steps. If the solver fails without reaching the
    threshold, the rest of the interval is filled with int_threshold.

    Parameters
    ----------
    f : callable
        f(x, t, θ) -> dx/dt, numpy, batched: x (m, d), t (m,) -> (m, d).
    t_loss : np.ndarray, shape (m,)
        Points in [t_0, t_1] at which to return the trajectory.
    **kwargs :
        Forwarded to `scipy.integrate.solve_ivp` (method, rtol, atol, max_step, ...).

    Returns
    -------
    x_loss : np.ndarray, shape (m, d)
        x(t_loss).
    x_end : np.ndarray, shape (d,)
        x(t_1), for the continuity constraint.
    """

    def rhs(t, y):
        return np.asarray(f(y[None, :], np.array([t]), theta), dtype=float)[0]

    def blow_up(t, y):
        return int_threshold - np.max(np.abs(y))

    blow_up.terminal = True

    sol = solve_ivp(rhs, (t_0, t_1), x0, events=blow_up, dense_output=True, **kwargs)

    t_query = np.append(t_loss, t_1)
    x = np.full((len(t_query), len(x0)), float(int_threshold))
    reached = t_query <= sol.t[-1]
    if reached.any() and sol.sol is not None:
        x[reached] = sol.sol(t_query[reached]).T
    if sol.status == 1:  # threshold event
        x[~reached] = sol.y_events[0][0]
    x = np.nan_to_num(x, nan=int_threshold, posinf=int_threshold, neginf=-int_threshold)
    return x[:-1], x[-1]


def _scipy(f, ts, xs, n_subintervals, theta_0, unobserved, unobserved_inits, int_threshold,
           minimizer_kwargs=None, **kwargs):
    """See `ms`. Finite-difference gradients, everything in numpy."""
    ts, xs, theta_0 = np.asarray(ts, float), np.asarray(xs, float), np.asarray(theta_0, float)
    n, d = xs.shape
    if not 1 <= n_subintervals <= n - 1:
        raise ValueError(f"n_subintervals must be between 1 and len(ts) - 1 = {n - 1}, got {n_subintervals}.")

    hidden = np.zeros(d, dtype=bool)
    if unobserved is not None:
        hidden[list(unobserved)] = True
    observed = ~hidden

    # Shooting nodes on data points, so every subinterval gets about the same number of points.
    # Subinterval k runs from ts[nodes[k]] to ts[nodes[k+1]] and is scored on the points
    # nodes[k] .. nodes[k+1] - 1 (the last one also on its end point), so every point counts once.
    nodes = np.linspace(0, n - 1, n_subintervals + 1).round().astype(int)
    scored = [slice(nodes[k], nodes[k + 1] + (k == n_subintervals - 1)) for k in range(n_subintervals)]

    # Initial node states: the data for observed dimensions, unobserved_inits for the others.
    # TODO: better default for unobserved dimensions than N(0, 1) (e.g. a short single-shooting
    #  fit, or physically plausible values); N(0, 1) can start positive-only systems negative.
    s_0 = xs[nodes[:-1]].copy()
    if hidden.any():
        guess = np.random.normal(0, 1, (n_subintervals, hidden.sum())) if unobserved_inits is None else unobserved_inits
        s_0[:, hidden] = np.broadcast_to(np.asarray(guess, float), (n_subintervals, hidden.sum()))
    z_0 = np.concatenate([s_0.ravel(), theta_0])

    def unpack(z):
        return z[: n_subintervals * d].reshape(n_subintervals, d), z[n_subintervals * d :]

    # One integration serves both the loss and the continuity defects at the same z. The finite
    # differences of the objective and of the constraint are taken at the same perturbed points,
    # so the cache must hold a whole gradient's worth of them.
    # TODO: exploit the block structure - subinterval k's residuals depend only on s_k and θ, so
    #  one perturbation of θ plus one per component of each s_k (integrating only subinterval k)
    #  gives the whole Jacobian, instead of integrating all subintervals for every component of z.
    @functools.lru_cache(maxsize=2 * (len(z_0) + 2))
    def shoot(key: bytes):
        s, theta = unpack(np.frombuffer(key))
        sse, defects = 0.0, []
        for k in range(n_subintervals):
            a, b = nodes[k], nodes[k + 1]
            x_loss, x_end = _scipy_trajectory(f, ts[a], ts[b], ts[scored[k]], s[k], theta, int_threshold, **kwargs)
            # only observed dimensions have data to compare with
            sse += float(((x_loss - xs[scored[k]])[:, observed] ** 2).sum())
            if k < n_subintervals - 1:
                defects.append(s[k + 1] - x_end)
        return sse, np.concatenate(defects) if defects else np.empty(0)

    def loss(z):
        return shoot(np.asarray(z, float).tobytes())[0]

    def continuity(z):
        return shoot(np.asarray(z, float).tobytes())[1]

    # TODO: a blown-up subinterval is constant in the parameters, so its finite-difference
    #  gradient is 0 there; a start far from any stable θ can stall.
    # TODO: per-dimension weights for the residuals, for states on very different scales.
    options = {"method": "trust-constr", "jac": "2-point", **(minimizer_kwargs or {})}
    if n_subintervals > 1:
        options.setdefault("constraints", [NonlinearConstraint(continuity, 0.0, 0.0)])
    result = scipy_minimize(loss, z_0, **options)

    result.inits, result.theta = unpack(result.x)
    result.nodes = ts[nodes[:-1]]
    return result


def ms(
    f,
    ts,
    xs,
    n_subintervals,
    theta_0,
    unobserved: None | Iterable = None,
    unobserved_inits =None,
    engine: Literal["torch", "scipy"] = "torch",
    int_threshold=1e6,
    minimizer_kwargs: "dict | None" = None,
    **kwargs,
):
    """
    θ-parameter  estimation for equation x'=F(x,t,θ) on datapoints (ts, xs) using multishooting. Allows unobserved variables.

    Time interval is divided into n_subintervals subintervals. Initial value of x is picked and optimized on the begining of every interval.
    For subintervals ts_1, ..., ts_n and initial values s_1, ..., s_n we minimize

    Σ_i || data(ts_i)[observed] - x(i, θ)(ts_i)[observed] ||², where x(i, θ)' = f(x(i, θ), t, θ) and x(i, θ)(ts_i[0]) = s_i

    on the manifold of such trajectories that x(i-1, θ)(ts_i[0]) = s_i. The node states s_i are
    optimized in every dimension, observed or not.


    Parameters
    ----------
    f : Callable
        f(x, t, θ) - gradient at time t, batched: x (m, d), t (m,) -> (m, d). numpy for
        engine="scipy".
    ts : array-like, shape (n,)
        timepoints, increasing
    xs : array-like, shape (n, d)
        true values of x at time t. Columns listed in `unobserved` are ignored (may be NaN).
    n_subintervals : int
        Number of subintervals for multishooting, 1 (single shooting) to n - 1. Nodes are placed
        on data points.
    theta_0 : array-like
        Initial guess for parameters θ.
    unobserved : Iterable | None, optional
        List (or other iterable) of unobserved dimensions (column indices) of x.
    unobserved_inits : array-like, optional
        Initial guesses for the unobserved dimensions at the nodes, shape (n_unobserved,) (same at
        every node) or (n_subintervals, n_unobserved). Default: N(0, 1) samples. Observed
        dimensions start from the data at the nodes.
    engine : "torch" or "scipy"
        engine to compute gradients. If torch is used, f must be torch-compatible and derivable.
    int_threshold : float | int
        Integration threshold. If trajectory reaches +-int_treshold, integrations stops and constant values are returned for further times
    minimizer_kwargs : dict, optional
        engine="scipy": forwarded to `scipy.optimize.minimize`, over the defaults
        method="trust-constr", jac="2-point" (e.g. {"options": {"gtol": 1e-6, "maxiter": 500}}).
    **kwargs :
        Integration settings, forwarded to `scipy.integrate.solve_ivp` (method, rtol, atol, ...).

    Returns
    -------
    scipy.optimize.OptimizeResult
        engine="scipy": plus `.theta`, `.inits` (n_subintervals, d) node states and `.nodes`
        (their times).
    """
    if engine == "torch":
        return _torch(f, ts, xs, n_subintervals, theta_0, unobserved, unobserved_inits, int_threshold, **kwargs)
    elif engine == "scipy":
        return _scipy(f, ts, xs, n_subintervals, theta_0, unobserved, unobserved_inits, int_threshold,
                      minimizer_kwargs=minimizer_kwargs, **kwargs)
    else:
        raise ValueError(f"{engine} is not a recognized engine. Use scipy or torch.")
