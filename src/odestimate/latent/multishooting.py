import functools
import warnings
from dataclasses import dataclass
from typing import Iterable, Literal
import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import NonlinearConstraint


FIXED_STEP_METHODS = ("euler", "rk4")


def _fixed_step(rhs, t_0, t_1, y0, t_query, method, step, int_threshold, d):
    """
    Fixed-step explicit integration of y' = rhs(t, y) on [t_0, t_1], with steps of at most `step`
    on the grid t_0, t_0 + step, ..., cut also at every point of t_query (so those are hit exactly).
    Stages stay inside the half-open step [t, t + h): an input that jumps at t + h (e.g. a daily
    value held on [t_i, t_{i+1})) is not read by this step. Once |y[:d]| reaches int_threshold the
    state is held, as with the adaptive solver.

    Returns y at every point of t_query, shape (len(t_query), len(y0)).
    """
    grid = np.unique(np.concatenate([np.arange(t_0, t_1, step), t_query, [t_0, t_1]]))
    grid = grid[(grid >= t_0) & (grid <= t_1)]
    ys = np.empty((len(grid), len(y0)))
    y, held = np.asarray(y0, float), False
    ys[0] = y
    for i in range(len(grid) - 1):
        if not held:
            t, h = grid[i], grid[i + 1] - grid[i]
            if method == "euler":
                y = y + h * rhs(t, y)
            else:
                k1 = rhs(t, y)
                k2 = rhs(t + h / 2, y + h / 2 * k1)
                k3 = rhs(t + h / 2, y + h / 2 * k2)
                k4 = rhs(np.nextafter(t + h, t), y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            held = not np.all(np.isfinite(y[:d])) or np.max(np.abs(y[:d])) >= int_threshold
        ys[i + 1] = y
    return ys[np.searchsorted(grid, t_query)]


def _integrate(f, t_0, t_1, t_query, x0s, thetas, int_threshold, batched_theta=False, **kwargs):
    """
    Integrates m copies of x' = f(x, t, θ) as one system: copy i from x0s[i] with θ = thetas[i].

    One system means one step sequence for all copies, so differences between copies are smooth
    in the perturbation that separates them (internal numerical differentiation). Differences of
    separate adaptive solves are not: each solve picks its own steps, and a perturbation of 1e-8
    changes the result by the solver tolerance, not by 1e-8 times the derivative.

    If |x| of copy 0 reaches int_threshold, integration stops and every copy holds the state it
    reached to t_1: a diverging trajectory costs a large, finite loss instead of overflowing or
    grinding on with ever smaller steps. If the solver fails, the rest is filled with int_threshold.

    Parameters
    ----------
    f : callable
        f(x, t, θ) -> dx/dt, numpy, batched: x (m, d), t (m,) -> (m, d).
    t_query : np.ndarray, shape (q,)
        Points in [t_0, t_1] at which to return the copies.
    x0s : np.ndarray, shape (m, d)
    thetas : np.ndarray, shape (m, p)
    batched_theta : bool
        f accepts θ of shape (m, p), one row per copy: all copies in one call of f. Otherwise one
        call per distinct θ.
    **kwargs :
        Forwarded to `scipy.integrate.solve_ivp` (method, rtol, atol, max_step, ...). Or
        method="euler" / "rk4" with step=h: fixed-step integration instead (`_fixed_step`).

    Returns
    -------
    x_query : np.ndarray, shape (m, q, d)
    x_end : np.ndarray, shape (m, d)
    """
    m, d = x0s.shape
    if batched_theta:
        groups = [(np.arange(m), thetas)]
    else:
        distinct, which = np.unique(thetas, axis=0, return_inverse=True)
        groups = [(np.flatnonzero(which.ravel() == g), theta) for g, theta in enumerate(distinct)]

    def rhs(t, y):
        x = y.reshape(m, d)
        dx = np.empty_like(x)
        for rows, theta in groups:
            dx[rows] = f(x[rows], np.full(len(rows), t), theta)
        return dx.ravel()

    def blow_up(t, y):
        return int_threshold - np.max(np.abs(y[:d]))

    blow_up.terminal = True

    t_all = np.append(t_query, t_1)
    if kwargs.get("method") in FIXED_STEP_METHODS:
        if "step" not in kwargs:
            raise ValueError(f"method={kwargs['method']!r} needs a step size, e.g. step=1.0")
        x = _fixed_step(rhs, t_0, t_1, x0s.ravel(), t_all, kwargs["method"], kwargs["step"], int_threshold, d)
    else:
        sol = solve_ivp(rhs, (t_0, t_1), x0s.ravel(), events=blow_up, dense_output=True, **kwargs)
        x = np.full((len(t_all), m * d), float(int_threshold))
        reached = t_all <= sol.t[-1]
        if reached.any() and sol.sol is not None:
            x[reached] = sol.sol(t_all[reached]).T
        if sol.status == 1:  # threshold event
            x[~reached] = sol.y_events[0][0]
    x = np.nan_to_num(x, nan=int_threshold, posinf=int_threshold, neginf=-int_threshold)
    x = x.reshape(len(t_all), m, d).transpose(1, 0, 2)
    return x[:, :-1], x[:, -1]


def _scipy_trajectory(f, t_0, t_1, t_loss, x0, theta, int_threshold, **kwargs):
    """
    Solves x' = f(x, t, θ) with x(t_0) = x0 on [t_0, t_1], for fixed θ. See `_integrate` for the
    behaviour at int_threshold and for **kwargs.

    Returns
    -------
    x_loss : np.ndarray, shape (len(t_loss), d)
        x(t_loss).
    x_end : np.ndarray, shape (d,)
        x(t_1).
    """
    x_query, x_end = _integrate(f, t_0, t_1, t_loss, np.asarray(x0, float)[None], np.asarray(theta, float)[None],
                                int_threshold, **kwargs)
    return x_query[0], x_end[0]


def _huber(r, delta):
    """
    Huber loss of the residuals r: r² for |r| <= delta, 2 delta |r| - delta² beyond (the same
    value and slope at |r| = delta, then linear).

    Returns
    -------
    loss : np.ndarray
        Per residual.
    slope : np.ndarray
        d loss / d r: 2r inside, bounded by ±2 delta outside.
    irls : np.ndarray
        slope / r: 2 inside, 2 delta / |r| outside - the weight of the residual in the Gauss-Newton
        Hessian J^T diag(irls) J (iteratively reweighted least squares). Positive everywhere, so
        the Hessian of the residuals beyond delta doesn't vanish, it only shrinks.
    """
    a = np.abs(r)
    inside = a <= delta
    loss = np.where(inside, r**2, 2 * delta * a - delta**2)
    slope = np.where(inside, 2 * r, 2 * delta * np.sign(r))
    irls = np.where(inside, 2.0, 2 * delta / np.maximum(a, delta))
    return loss, slope, irls


@dataclass
class _Layout:
    """Data, weights and subinterval layout shared by both engines (see `_layout`)."""

    ts: np.ndarray
    xs: np.ndarray  # (n, d) state values, NaN where unknown - for the initial node states
    theta_0: np.ndarray
    d: int
    p: int
    n_subintervals: int
    hidden: np.ndarray  # (d,) bool: no value in xs at the first node
    ys: np.ndarray  # (n, q) observed data, 0 where missing
    mask: np.ndarray  # (n, q) bool: where ys has data
    weights: np.ndarray  # (q,)
    observe: "object"  # observe(x, t, θ) -> (m, q); None: the state itself
    nodes: np.ndarray  # (n_subintervals + 1,) indices into ts
    scored: list  # slice of ts per subinterval

    def unpack(self, z):
        k = self.n_subintervals * self.d
        return z[:k].reshape(self.n_subintervals, self.d), z[k:]


def _layout(ts, xs, n_subintervals, theta_0, unobserved, weights, observe=None, ys=None) -> _Layout:
    ts, xs, theta_0 = np.asarray(ts, float), np.array(xs, float), np.asarray(theta_0, float)
    n, d = xs.shape
    if not 1 <= n_subintervals <= n - 1:
        raise ValueError(f"n_subintervals must be between 1 and len(ts) - 1 = {n - 1}, got {n_subintervals}.")
    if unobserved is not None:
        xs[:, list(unobserved)] = np.nan
    if observe is None:
        ys = xs
    elif ys is None:
        raise ValueError("observe needs the observed data ys, shape (len(ts), number of observed quantities).")
    ys = np.asarray(ys, float).reshape(n, -1)
    mask = np.isfinite(ys)

    # Residuals are divided by the mean magnitude of their variable, so every observed variable
    # counts in relative terms and a variable in large units doesn't drown the others. The Huber
    # threshold is on this scale too: huber_delta = 1 means "off by as much as the variable's
    # mean". A variable whose mean is near 0 but that varies a lot would get a huge weight - then
    # pass `weights` (e.g. 1 / noise std).
    if weights is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            scale = np.nanmean(np.abs(ys), axis=0)
        weights = 1.0 / np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

    # Shooting nodes on data points, so every subinterval gets about the same number of points.
    # Subinterval k runs from ts[nodes[k]] to ts[nodes[k+1]] and is scored on the points
    # nodes[k] .. nodes[k+1] - 1 (the last one also on its end point), so every point counts once.
    nodes = np.linspace(0, n - 1, n_subintervals + 1).round().astype(int)
    scored = [slice(nodes[k], nodes[k + 1] + (k == n_subintervals - 1)) for k in range(n_subintervals)]
    return _Layout(ts, xs, theta_0, d, len(theta_0), n_subintervals, np.isnan(xs[0]),
                   np.where(mask, ys, 0.0), mask, np.asarray(weights, float), observe, nodes, scored)


def _initial_nodes(lay: _Layout, unobserved_inits, end_of_subinterval, int_threshold) -> np.ndarray:
    """
    Initial node states: xs at the nodes where it has a value. The dimensions without a value at
    the first node ("hidden") start from unobserved_inits (N(0, 1) samples if None) there, and
    every missing value at a later node is carried forward: integrate subinterval k from node k with
    θ_0 (`end_of_subinterval(k, s_k) -> x(t_{k+1})`) and take where it ended. Those parts of the
    continuity constraint are then 0 from the start, and the optimizer only has to close the gaps
    where xs has values (data vs. integration). A 2-D unobserved_inits (one row per node) is used
    as given for the hidden dimensions.
    """
    s_0 = lay.xs[lay.nodes[:-1]].copy()
    hidden = lay.hidden
    if hidden.any():
        if unobserved_inits is not None and np.ndim(unobserved_inits) == 2:
            s_0[:, hidden] = np.asarray(unobserved_inits, float)
        else:
            s_0[0, hidden] = np.random.normal(0, 1, hidden.sum()) if unobserved_inits is None else unobserved_inits
    for k in range(lay.n_subintervals - 1):
        missing = np.isnan(s_0[k + 1])
        if missing.any():
            x_end = end_of_subinterval(k, s_0[k])
            blew_up = not np.all(np.isfinite(x_end)) or np.max(np.abs(x_end)) >= int_threshold
            s_0[k + 1, missing] = (s_0[k] if blew_up else x_end)[missing]
    return np.concatenate([s_0.ravel(), lay.theta_0])


def _gauss_newton_sqp(evaluate, linearize, z_0, lay: _Layout, huber_delta, minimizer_kwargs):
    """
    Minimizes the Huber loss of the residuals under the continuity constraint, with trust-constr
    and a Gauss-Newton model. `evaluate(z) -> (r, c)` gives residuals and continuity defects,
    `linearize(z) -> (r, J_r, c, J_c)` also their Jacobians; both are cached per z, since
    trust-constr asks for the objective and the constraint (and their derivatives) at the same z.
    """
    n_z = len(z_0)
    key = lambda z: np.asarray(z, float).tobytes()
    evaluate_z, linearize_z = evaluate, linearize
    evaluate = functools.lru_cache(maxsize=8)(lambda k: evaluate_z(np.frombuffer(k)))
    linearize = functools.lru_cache(maxsize=8)(lambda k: linearize_z(np.frombuffer(k)))

    def loss(z):
        return float(_huber(evaluate(key(z))[0], huber_delta)[0].sum())

    def gradient(z):
        r, J_r, _, _ = linearize(key(z))
        return J_r.T @ _huber(r, huber_delta)[1]

    def gauss_newton_hessian(z):
        r, J_r, _, _ = linearize(key(z))
        return J_r.T @ (_huber(r, huber_delta)[2][:, None] * J_r)

    # The Gauss-Newton model also drops the constraints' curvature (their linearization is kept):
    # the SQP step then solves a linear least-squares problem under linearized continuity, which is
    # what classical multiple shooting (Bock's generalized Gauss-Newton) does.
    constraints = NonlinearConstraint(
        lambda z: evaluate(key(z))[1], 0.0, 0.0,
        jac=lambda z: linearize(key(z))[3],
        hess=lambda z, v: np.zeros((n_z, n_z)),
    )

    # TODO: a blown-up subinterval is held constant, so it is flat in the parameters: Huber bounds
    #  its pull on the step, but gives it no slope back toward a stable θ. A start far from any
    #  stable θ can still stall.
    # TODO: the trust region is in raw z, where node states (data units) and θ can differ by
    #  orders of magnitude; trust-constr has no x_scale, so rescale z by hand if steps stall.
    options = {"method": "trust-constr", "jac": gradient, "hess": gauss_newton_hessian, **(minimizer_kwargs or {})}
    if lay.n_subintervals > 1:
        options.setdefault("constraints", [constraints])
    result = scipy_minimize(loss, z_0, **options)

    result.inits, result.theta = lay.unpack(result.x)
    result.nodes = lay.ts[lay.nodes[:-1]]
    result.weights = lay.weights
    return result


def _scipy(f, ts, xs, n_subintervals, theta_0, unobserved, unobserved_inits, int_threshold,
           minimizer_kwargs=None, weights=None, huber_delta=1.0, batched_theta=False, observe=None, ys=None,
           **kwargs):
    """See `ms`. Jacobian by forward differences, block by block; everything in numpy."""
    lay = _layout(ts, xs, n_subintervals, theta_0, unobserved, weights, observe, ys)
    ts, d, p, nodes, scored = lay.ts, lay.d, lay.p, lay.nodes, lay.scored
    n_z = n_subintervals * d + p
    g = observe if observe is not None else (lambda x, t, theta: x)

    def end_of_subinterval(k, s_k):
        return _scipy_trajectory(f, ts[nodes[k]], ts[nodes[k + 1]], np.empty(0), s_k, lay.theta_0, int_threshold, **kwargs)[1]

    z_0 = _initial_nodes(lay, unobserved_inits, end_of_subinterval, int_threshold)

    def residual(k, x_loss, theta):
        """Weighted residuals of subinterval k, 0 where there is no data. x_loss (m, d) with one θ,
        or (copies, m, d) with one θ per copy."""
        t_k = ts[scored[k]]
        if x_loss.ndim == 2:
            y = g(x_loss, t_k, theta)
        elif batched_theta:
            c, m = x_loss.shape[:2]
            y = g(x_loss.reshape(c * m, d), np.tile(t_k, c), np.repeat(theta, m, axis=0)).reshape(c, m, -1)
        else:
            y = np.stack([g(x, t_k, th) for x, th in zip(x_loss, theta)])
        return np.where(lay.mask[scored[k]], (y - lay.ys[scored[k]]) * lay.weights, 0.0).reshape(y.shape[:-2] + (-1,))

    def evaluate(z):
        """Residuals and continuity defects at z (plain integrations)."""
        s, theta = lay.unpack(z)
        residuals, defects = [], []
        for k in range(n_subintervals):
            a, b = nodes[k], nodes[k + 1]
            x_loss, x_end = _scipy_trajectory(f, ts[a], ts[b], ts[scored[k]], s[k], theta, int_threshold, **kwargs)
            residuals.append(residual(k, x_loss, theta))
            if k < n_subintervals - 1:
                defects.append(s[k + 1] - x_end)
        return np.concatenate(residuals), np.concatenate(defects) if defects else np.empty(0)

    def linearize(z):
        """
        Residuals and defects with their Jacobians, by forward differences, block by block.
        Subinterval k's residuals and end point depend only on its node state s_k and on θ, so it is
        integrated once, with 1 + d + p copies side by side (see `_integrate`): unperturbed, each
        component of s_k perturbed, each component of θ perturbed. That's n_subintervals
        integrations per Jacobian instead of one full pass over all subintervals per component of z.
        """
        s, theta = lay.unpack(z)
        h_s, h_theta = lay.unpack(np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(z)))
        J_r = np.zeros((lay.ys.size, n_z))
        J_c = np.zeros(((n_subintervals - 1) * d, n_z))
        residuals, defects = [], []
        row = 0
        for k in range(n_subintervals):
            a, b = nodes[k], nodes[k + 1]
            x0s = np.vstack([s[k], s[k] + np.diag(h_s[k]), np.repeat(s[k][None], p, 0)])
            thetas = np.vstack([np.repeat(theta[None], 1 + d, 0), theta + np.diag(h_theta)])
            x_loss, x_end = _integrate(f, ts[a], ts[b], ts[scored[k]], x0s, thetas, int_threshold,
                                       batched_theta=batched_theta, **kwargs)
            r = residual(k, x_loss, thetas)
            r_k = r[0]
            rows = slice(row, row + len(r_k))
            columns = np.r_[k * d : (k + 1) * d, n_subintervals * d : n_z]
            steps = np.r_[h_s[k], h_theta]
            J_r[rows, columns] = ((r[1:] - r_k) / steps[:, None]).T
            residuals.append(r_k)
            row += len(r_k)
            if k < n_subintervals - 1:
                c_rows = slice(k * d, (k + 1) * d)
                J_c[c_rows, columns] = -((x_end[1:] - x_end[0]) / steps[:, None]).T
                J_c[c_rows, (k + 1) * d : (k + 2) * d] = np.eye(d)
                defects.append(s[k + 1] - x_end[0])
        return np.concatenate(residuals), J_r, np.concatenate(defects) if defects else np.empty(0), J_c

    return _gauss_newton_sqp(evaluate, linearize, z_0, lay, huber_delta, minimizer_kwargs)


def _torch_euler(f, x0, grid, steps, theta, int_threshold):
    """
    Explicit Euler for a batch of independent rows at once: row i starts at x0[i] and steps over
    grid[i] (steps[i] = its step sizes; 0 for padding). Every step is one call of f for all rows.
    A row whose |x| would reach int_threshold (or turn non-finite) is held from there, at the
    clipped value, like the scipy engine.

    Returns the states at every grid point, shape (len(grid[0]), m, d).
    """
    x, alive, states = x0, torch.ones(len(x0), dtype=torch.bool), [x0]
    for j in range(grid.shape[1] - 1):
        x_new = x + steps[:, j, None] * f(x, grid[:, j], theta)
        ok = x_new.detach().abs().amax(1) < int_threshold  # False for NaN too
        held = torch.nan_to_num(x_new, nan=int_threshold, posinf=int_threshold, neginf=-int_threshold)
        held = held.clamp(-int_threshold, int_threshold)
        x = torch.where(alive[:, None], torch.where(ok[:, None], x_new, held), x)
        alive = alive & ok
        states.append(x)
    return torch.stack(states)


def _torch(f, ts, xs, n_subintervals, theta_0, unobserved, unobserved_inits, int_threshold,
           minimizer_kwargs=None, weights=None, huber_delta=1.0, observe=None, ys=None, **kwargs):
    """
    See `ms`. Fixed-step explicit Euler, every subinterval integrated at once (one call of f per
    step, for all subintervals together), Jacobian by reverse-mode autodiff. The optimizer is the
    same Gauss-Newton SQP as engine="scipy".
    """
    unknown = set(kwargs) - {"method", "step"}
    if unknown or kwargs.get("method", "euler") != "euler":
        raise ValueError(f"engine='torch' integrates with method='euler' and a fixed step (default 1); got {kwargs}.")
    step = kwargs.get("step", 1.0)

    lay = _layout(ts, xs, n_subintervals, theta_0, unobserved, weights, observe, ys)
    ts, d, nodes = lay.ts, lay.d, lay.nodes
    g = observe if observe is not None else (lambda x, t, theta: x)
    as_t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)

    # Each subinterval's step grid: every `step` from its start, cut also at its data points (as
    # in `_fixed_step`), padded with zero steps to a common length so all of them step together.
    grids = []
    for k in range(n_subintervals):
        a, b = ts[nodes[k]], ts[nodes[k + 1]]
        points = np.unique(np.concatenate([np.arange(a, b, step), ts[lay.scored[k]], [a, b]]))
        grids.append(points[(points >= a) & (points <= b)])
    length = max(len(points) for points in grids)
    grid = np.stack([np.pad(points, (0, length - len(points)), mode="edge") for points in grids])
    steps = np.diff(grid, axis=1)
    # scored data points as (grid position, subinterval) pairs, in data order
    position = np.concatenate([np.searchsorted(grids[k], ts[lay.scored[k]]) for k in range(n_subintervals)])
    subinterval = np.concatenate([np.full(s.stop - s.start, k) for k, s in enumerate(lay.scored)])
    grid_t, steps_t, ts_t = as_t(grid), as_t(steps), as_t(ts)
    ys_t, weights_t, mask_t = as_t(lay.ys), as_t(lay.weights), torch.as_tensor(lay.mask)

    def shoot(z: torch.Tensor):
        s, theta = lay.unpack(z)
        states = _torch_euler(f, s, grid_t, steps_t, theta, int_threshold)
        y = g(states[position, subinterval], ts_t, theta)
        residuals = torch.where(mask_t, (y - ys_t) * weights_t, 0.0).reshape(-1)
        defects = (s[1:] - states[-1, :-1]).reshape(-1)
        return torch.cat([residuals, defects]), residuals.numel()

    def end_of_subinterval(k, s_k):
        with torch.no_grad():
            states = _torch_euler(f, as_t(s_k)[None], grid_t[k : k + 1], steps_t[k : k + 1],
                                  as_t(lay.theta_0), int_threshold)
        return states[-1, 0].numpy()

    z_0 = _initial_nodes(lay, unobserved_inits, end_of_subinterval, int_threshold)

    def evaluate(z):
        with torch.no_grad():
            out, n_r = shoot(as_t(z))
        out = out.numpy()
        return out[:n_r], out[n_r:]

    def linearize(z):
        z_t = as_t(z)
        jac = torch.func.jacrev(lambda z_: shoot(z_)[0])(z_t)
        with torch.no_grad():
            out, n_r = shoot(z_t)
        out, jac = out.numpy(), torch.nan_to_num(jac).numpy()
        return out[:n_r], jac[:n_r], out[n_r:], jac[n_r:]

    return _gauss_newton_sqp(evaluate, linearize, z_0, lay, huber_delta, minimizer_kwargs)


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
    weights=None,
    huber_delta: float = 1.0,
    batched_theta: bool = False,
    observe=None,
    ys=None,
    **kwargs,
):
    """
    θ-parameter  estimation for equation x'=F(x,t,θ) on datapoints (ts, xs) using multishooting. Allows unobserved variables.

    Time interval is divided into n_subintervals subintervals. Initial value of x is picked and optimized on the begining of every interval.
    For subintervals ts_1, ..., ts_n and initial values s_1, ..., s_n we minimize

    Σ_i Σ_j huber( w_j (data(ts_i)[j] - g(x(i, θ)(ts_i), ts_i, θ)[j]) ) over the data that exists,
    where x(i, θ)' = f(x(i, θ), t, θ) and x(i, θ)(ts_i[0]) = s_i

    g is the observation: the state itself by default (data = xs), or any function of the state
    (`observe`, data = ys), e.g. an output computed from the states.

    on the manifold of such trajectories that x(i-1, θ)(ts_i[0]) = s_i. The node states s_i are
    optimized in every dimension, observed or not.


    Parameters
    ----------
    f : Callable
        f(x, t, θ) - gradient at time t, batched: x (m, d), t (m,) -> (m, d). numpy for
        engine="scipy"; torch for engine="torch", differentiable in x and θ (it is called with
        x of shape (n_subintervals, d): every subinterval in one call).
    ts : array-like, shape (n,)
        timepoints, increasing
    xs : array-like, shape (n, d)
        true values of x at time t, NaN where unknown. Columns listed in `unobserved` are ignored.
        Without `observe` this is the data; with it, xs only seeds the node states (may be all NaN).
    n_subintervals : int
        Number of subintervals for multishooting, 1 (single shooting) to n - 1. Nodes are placed
        on data points.
    theta_0 : array-like
        Initial guess for parameters θ.
    unobserved : Iterable | None, optional
        List (or other iterable) of unobserved dimensions (column indices) of x.
    unobserved_inits : array-like, optional
        Initial guess for the dimensions with no value in xs at ts[0]. Shape (n_unobserved,): the
        state at the first node; every missing node value later is carried forward by integrating
        with θ_0. Shape (n_subintervals, n_unobserved): one row per node, used as given. Default:
        N(0, 1) samples at the first node. The other dimensions start from xs at the nodes.
    engine : "torch" or "scipy"
        engine to compute gradients. "scipy": any integrator (see **kwargs), Jacobian by forward
        differences. "torch": fixed-step explicit Euler only, Jacobian by autodiff; f must be
        torch-compatible and derivable. Both use the same optimizer.
    int_threshold : float | int
        Integration threshold. If trajectory reaches +-int_treshold, integrations stops and constant values are returned for further times
    minimizer_kwargs : dict, optional
        Forwarded to `scipy.optimize.minimize`, over the defaults
        method="trust-constr" with the Gauss-Newton gradient and Hessian
        (e.g. {"options": {"gtol": 1e-6, "maxiter": 500}}).
    weights : array-like, shape (n_observed,), optional
        Residual weight per observed dimension. Default: 1 / mean |data| of that dimension.
    huber_delta : float
        Weighted residuals up to huber_delta count quadratically, beyond it linearly (Huber). Bounds
        the influence of outliers and of blown-up subintervals (see `_huber`).
    batched_theta : bool
        f accepts θ of shape (m, p), one row per x row. Then the Jacobian's perturbed copies of a
        subinterval are evaluated in a single call of f (engine="scipy").
    observe : Callable, optional
        g(x, t, θ) -> (m, q): what the data measures, batched like f (and with θ of shape (m, p)
        when batched_theta=True). torch for engine="torch".
    ys : array-like, shape (n, q), optional
        The data for `observe`, NaN where missing.
    **kwargs :
        Integration settings. engine="scipy": forwarded to `scipy.integrate.solve_ivp` (method,
        rtol, atol, ...), or method="euler" / "rk4" with step=h for a fixed step instead: much
        cheaper when inputs jump on a regular grid (e.g. daily data), which an adaptive solver
        resolves with many small steps. step=1 with method="euler" on daily data is a daily
        difference model. engine="torch": only step=h (default 1), always Euler.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Plus `.theta`, `.inits` (n_subintervals, d) node states, `.nodes` (their times) and
        `.weights`.
    """
    if engine == "torch":
        return _torch(f, ts, xs, n_subintervals, theta_0, unobserved, unobserved_inits, int_threshold,
                      minimizer_kwargs=minimizer_kwargs, weights=weights, huber_delta=huber_delta,
                      observe=observe, ys=ys, **kwargs)
    elif engine == "scipy":
        return _scipy(f, ts, xs, n_subintervals, theta_0, unobserved, unobserved_inits, int_threshold,
                      minimizer_kwargs=minimizer_kwargs, weights=weights, huber_delta=huber_delta,
                      batched_theta=batched_theta, observe=observe, ys=ys, **kwargs)
    else:
        raise ValueError(f"{engine} is not a recognized engine. Use scipy or torch.")
