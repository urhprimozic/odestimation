"""
Collocation: θ-parameter estimation for x' = f(x, t, θ) as the constrained problem

    argmin_{U, θ}  loss(U, θ)   subject to   g(U, θ) = 0.

U are the states at the seeds; between seeds the trajectory is linear in t.
    loss   compares the data at ts with the trajectory there (or with an observation of it):
           0.5 Σ ((observation - data) / σ)², i.e. -log p(data | U, θ) for Gaussian noise
           (0.5 Σ |(observation - data) / σ|^power with `power`).
    g      the slope of the trajectory minus f, at the ODE points: g = 0 makes the trajectory
           solve the ODE.
No integrator, and no GP: the states are unknowns, the ODE a constraint on them.

Counting. With n seeds and m ODE points per component, the manifold g = 0 has dimension
p + D (n - m). The default m = n - 1, one ODE point in the middle of every gap between seeds,
leaves exactly θ and the initial state to the data: g = 0 is then the implicit midpoint rule,
x_{k+1} = x_k + h f((x_k + x_{k+1}) / 2), an integrator. More ODE points take freedom from the
data and hand it to the discretization; an ODE point at the start of a gap gives explicit Euler.
"""

import functools
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, NonlinearConstraint, OptimizeResult, minimize

from odestimate.gradient_matching import powered
from odestimate.latent.magi import _pointwise_derivatives


def _gap(points, seeds):
    """Index k of the gap [seeds[k], seeds[k+1]) every point lies in. A point on a seed belongs to
    the gap that starts there, points outside the seeds to the first or last gap."""
    return np.clip(np.searchsorted(seeds, points, side="right") - 1, 0, len(seeds) - 2)


def _linear_interpolation(points, seeds):
    """
    Sparse matrices V, S of shape (len(points), len(seeds)): for values u at the seeds, V @ u are
    the values at `points` of the piecewise linear function through (seeds, u), S @ u its slopes.
    Outside the seeds the first and last pieces continue.
    """
    k = _gap(points, seeds)
    h = seeds[k + 1] - seeds[k]
    w = (points - seeds[k]) / h
    rows = np.repeat(np.arange(len(points)), 2)
    cols = np.stack([k, k + 1], axis=1).ravel()
    shape = (len(points), len(seeds))
    V = sparse.csr_matrix((np.stack([1 - w, w], axis=1).ravel(), (rows, cols)), shape=shape)
    S = sparse.csr_matrix((np.stack([-1 / h, 1 / h], axis=1).ravel(), (rows, cols)), shape=shape)
    return V, S


def _block_diagonal(blocks):
    """Sparse block-diagonal matrix with one (a, b) block per point, from blocks of shape (m, a, b)."""
    m, a, b = blocks.shape
    rows = np.broadcast_to((np.arange(m)[:, None] * a + np.arange(a))[:, :, None], blocks.shape)
    cols = np.broadcast_to((np.arange(m)[:, None] * b + np.arange(b))[:, None, :], blocks.shape)
    return sparse.csr_matrix((blocks.ravel(), (rows.ravel(), cols.ravel())), shape=(m * a, m * b))


def _cached(fn):
    """fn(z), remembered for the last few z: trust-constr asks for the loss, the constraint and
    their derivatives in separate calls at the same z."""
    cache = functools.lru_cache(maxsize=8)(lambda key: fn(np.frombuffer(key)))
    return lambda z: cache(np.asarray(z, float).tobytes())


class _Problem:
    """
    Loss residuals and constraint g in z = (U.ravel(), θ), U (n_seeds, D) the states at the seeds.
    Row-major everywhere: the state of component d at seed k is z[k D + d], and the states X (m, D)
    at m points are V @ U, flattened by kron(V, I_D) @ U.ravel().
    """

    def __init__(self, f, ts, seeds, ode_points, ys, sigma, observe, D, p, power=2.0):
        self.f, self.observe, self.ts, self.ode_points, self.power = f, observe, ts, ode_points, power
        self.n, self.D, self.p = len(seeds), D, p
        self.V_data, _ = _linear_interpolation(ts, seeds)
        self.V_ode, self.S_ode = _linear_interpolation(ode_points, seeds)
        self.mask = np.isfinite(ys)
        self.ys = np.where(self.mask, ys, 0.0)
        self.sigma = sigma

    def unpack(self, z):
        return z[: self.n * self.D].reshape(self.n, self.D), z[self.n * self.D :]

    def _observation(self, X, theta):
        if self.observe is None:
            return X
        return np.asarray(self.observe(X, self.ts, theta), float).reshape(len(self.ts), -1)

    def _f(self, X, theta):
        return np.asarray(self.f(X, self.ode_points, theta), float).reshape(len(self.ode_points), self.D)

    def evaluate(self, z):
        """(r, g): the loss residuals where there is data, and g at every ODE point and component."""
        U, theta = self.unpack(z)
        r = ((self._observation(self.V_data @ U, theta) - self.ys) / self.sigma)[self.mask]
        g = (self.S_ode @ U - self._f(self.V_ode @ U, theta)).ravel()
        return powered(r, self.power), g

    def linearize(self, z):
        """(r, J_r, g, J_g), the Jacobians sparse; d/dx and d/dθ of f and of the observation by
        forward differences."""
        U, theta = self.unpack(z)
        identity = sparse.identity(self.D, format="csr")

        X = self.V_data @ U
        Y = self._observation(X, theta)
        if self.observe is None:
            Yx, Yt = np.broadcast_to(np.eye(self.D), (len(self.ts), self.D, self.D)), np.zeros((len(self.ts), self.D, self.p))
        else:
            Yx, Yt = _pointwise_derivatives(self.observe, X, self.ts, theta, Y)
        per_channel = 1 / self.sigma[None, :, None]
        J_r = sparse.hstack([
            _block_diagonal(Yx * per_channel) @ sparse.kron(self.V_data, identity),
            sparse.csr_matrix((Yt * per_channel).reshape(-1, self.p)),
        ], format="csr")[self.mask.ravel()]
        r = ((Y - self.ys) / self.sigma)[self.mask]
        if self.power != 2:  # ρ = sign(r) |r|^(power/2), dρ/dr = power/2 |r|^(power/2 - 1)
            J_r = sparse.diags(self.power / 2 * np.abs(r) ** (self.power / 2 - 1)) @ J_r
            r = powered(r, self.power)

        X = self.V_ode @ U
        F = self._f(X, theta)
        Fx, Ft = _pointwise_derivatives(self.f, X, self.ode_points, theta, F)
        J_g = sparse.hstack([
            sparse.kron(self.S_ode, identity) - _block_diagonal(Fx) @ sparse.kron(self.V_ode, identity),
            sparse.csr_matrix(-Ft.reshape(-1, self.p)),
        ], format="csr")
        g = (self.S_ode @ U - F).ravel()
        return r, J_r, g, J_g


def _points(ts, n_seeds, seeds, n_ode_points, ode_points):
    """Seeds and ODE points from the arguments of `collocation`, see there."""
    if seeds is None:
        seeds = ts if n_seeds is None else np.linspace(ts[0], ts[-1], n_seeds)
    seeds = np.asarray(seeds, float)
    if ode_points is None and n_ode_points is None:
        ode_points = (seeds[:-1] + seeds[1:]) / 2
    elif ode_points is None:
        width = (seeds[-1] - seeds[0]) / n_ode_points
        ode_points = seeds[0] + width * (np.arange(n_ode_points) + 0.5)
    ode_points = np.asarray(ode_points, float)

    per_gap = np.bincount(_gap(ode_points, seeds), minlength=len(seeds) - 1)
    if np.any(per_gap != 1):
        warnings.warn(
            f"{np.sum(per_gap > 1)} gaps between seeds have more than one ODE point and {np.sum(per_gap == 0)} "
            "have none. The trajectory is linear in a gap, so a second ODE point there over-constrains it "
            "(the discretization, not the data, then fixes part of θ and the initial state), and in a gap "
            "without one the states follow the data, not the ODE. One ODE point per gap is the default."
        )
    return seeds, ode_points


def _initial_states(f, ts, xs, seeds, theta_0, unobserved_inits):
    """
    States at the seeds to start from: components with data from their data (linear interpolation),
    the others from unobserved_inits - a trajectory on ts, or their state at ts[0], carried over the
    seeds with θ_0 by explicit Euler.
    """
    D = xs.shape[1]
    hidden = [d for d in range(D) if not np.isfinite(xs[:, d]).any()]
    U = np.zeros((len(seeds), D))
    for d in range(D):
        if d not in hidden:
            ok = np.isfinite(xs[:, d])
            U[:, d] = np.interp(seeds, ts[ok], xs[ok, d])
    if not hidden:
        return U

    inits = np.zeros(len(hidden)) if unobserved_inits is None else np.asarray(unobserved_inits, float)
    if inits.ndim == 2:
        for j, d in enumerate(hidden):
            U[:, d] = np.interp(seeds, ts, inits[:, j])
        return U
    U[0, hidden] = inits
    for k in range(len(seeds) - 1):
        step = U[k] + (seeds[k + 1] - seeds[k]) * np.asarray(f(U[k : k + 1], seeds[k : k + 1], theta_0), float)[0]
        U[k + 1, hidden] = step[hidden] if np.all(np.isfinite(step)) else U[k, hidden]
    return U


def _clip(theta, bounds):
    if bounds is None:
        return theta
    return np.clip(theta, *(np.broadcast_to(np.asarray(b, float), theta.shape) for b in bounds))


def _minimize(evaluate, linearize, U_0, theta_0, bounds, kwargs):
    """
    Minimizes ½ |r(z)|² subject to g(z) = 0 over z = (U.ravel(), θ), from U_0 (n, D) and θ_0, with
    trust-constr and a Gauss-Newton model: `evaluate(z) -> (r, g)`, `linearize(z) -> (r, J_r, g, J_g)`.
    θ within `bounds`, the states free; no constraint if g is empty. `kwargs` for `minimize`.
    Returns minimize's result with .x in z.
    """
    lo, hi = (-np.inf, np.inf) if bounds is None else bounds
    lo, hi = (np.broadcast_to(np.asarray(b, float), theta_0.shape) for b in (lo, hi))
    free = np.full(U_0.size, np.inf)
    z_lo, z_hi = np.r_[-free, lo], np.r_[free, hi]

    # trust-constr's trust region is a ball in its variables and it has no x_scale: states in the
    # hundreds and a rate of 0.1 would share one radius. So it works in w = z / scale, the scale of
    # every state its component's mean |start|, of every parameter its |start|.
    typical = lambda a: np.where(a > 0, a, 1.0)
    scale = np.r_[np.tile(typical(np.abs(U_0).mean(axis=0)), len(U_0)), typical(np.abs(theta_0))]
    S = sparse.diags(scale)

    def loss(w):
        r, _ = evaluate(scale * w)
        return 0.5 * float(r @ r)

    def gradient(w):
        r, J_r, _, _ = linearize(scale * w)
        return scale * (J_r.T @ r)

    def gauss_newton_hessian(w):
        _, J_r, _, _ = linearize(scale * w)
        J = J_r @ S
        return (J.T @ J).tocsr()

    # Gauss-Newton also drops the curvature of g (its linearization is kept), as in multiple shooting.
    z_0 = np.r_[U_0.ravel(), theta_0]
    constraints = []
    if len(evaluate(z_0)[1]):
        constraints.append(NonlinearConstraint(lambda w: evaluate(scale * w)[1], 0.0, 0.0,
                                               jac=lambda w: linearize(scale * w)[3] @ S,
                                               hess=lambda w, v: sparse.csr_matrix((len(scale), len(scale)))))
    # trust-constr's default tolerances (1e-8) are below the noise of the forward-difference
    # derivatives (~1e-6): it would stop only at maxiter.
    kwargs = dict(kwargs)
    options = {"gtol": 1e-4, "barrier_tol": 1e-4, **kwargs.pop("options", {})}
    defaults = {"method": "trust-constr", "jac": gradient, "hess": gauss_newton_hessian, "constraints": constraints,
                "options": options}
    if bounds is not None:
        defaults["bounds"] = Bounds(z_lo / scale, z_hi / scale)
    result = minimize(loss, z_0 / scale, **{**defaults, **kwargs})
    result.x = scale * result.x
    return result


def collocation(f, ts, xs, theta_0, unobserved=None, unobserved_inits=None, n_seeds=None, seeds=None,
                n_ode_points=None, ode_points=None, observe=None, ys=None, sigma=None, bounds=None, power=2.0, **kwargs):
    """
    θ-parameter estimation for x' = f(x, t, θ): minimize the loss on the data subject to g = 0
    (see the module docstring), with trust-constr and a Gauss-Newton model of the loss.

    Parameters
    ----------
    f : callable
        f(x, t, θ) -> dx/dt, numpy, batched: x (m, D), t (m,) -> (m, D). Must act pointwise (row i
        of the output depends only on row i of x).
    ts : array-like, shape (n,)
        Data times, increasing. The loss is evaluated here.
    xs : array-like, shape (n, D)
        Data on the states, NaN where missing; a column without data is an unobserved component.
        With `observe`, xs only gives starting values (may be all NaN) and the data is `ys`.
    theta_0 : array-like, shape (p,)
        Initial guess for θ.
    unobserved : iterable of int, optional
        Columns of xs to treat as unobserved.
    unobserved_inits : array-like, optional
        Start for the unobserved components: shape (n_unobserved,), their state at ts[0], carried
        over the seeds with θ_0 by explicit Euler; or (len(ts), n_unobserved), a trajectory on ts.
        Default: state 0 at ts[0].
    n_seeds : int, optional
        Number of seeds, evenly spaced over ts. The unknowns are n_seeds * D + p.
    seeds : array-like, optional
        The seeds themselves, increasing (instead of n_seeds). Default: ts.
    n_ode_points : int, optional
        Number of ODE points, at the centres of that many equal parts of [seeds[0], seeds[-1]].
    ode_points : array-like, optional
        The ODE points themselves (instead of n_ode_points). Default: the midpoints between
        consecutive seeds (n_seeds - 1 of them).
    observe : callable, optional
        observe(x, t, θ) -> (m, q): what the data measures, batched and pointwise like f.
    ys : array-like, shape (n, q), optional
        The data for `observe`, NaN where missing.
    sigma : array-like, shape (q,), optional
        Noise std per data channel. Only the ratios matter for the estimate. Default: the std of
        each channel's data.
    bounds : (lo, hi), optional
        Bounds on θ.
    power : float
        Exponent of the data term, ½ Σ |r_i|^power (r_i the residuals divided by σ): 2 is least
        squares, a larger one punishes large deviations more.
    **kwargs :
        Forwarded to `scipy.optimize.minimize`, over the defaults method="trust-constr", the
        Gauss-Newton gradient and Hessian and the constraint g = 0. `options` are merged over
        gtol = barrier_tol = 1e-4 (e.g. options={"maxiter": 500}).

    Returns
    -------
    scipy.optimize.OptimizeResult
        minimize's result plus .theta, .x (n_seeds, D) the states at .seeds, .x_ts (n, D) the
        states at ts, .ode_points, .sigma and .cost (the loss at the optimum).
    """
    ts, xs, theta_0 = np.asarray(ts, float), np.array(xs, float), np.asarray(theta_0, float)
    if unobserved is not None:
        xs[:, list(unobserved)] = np.nan
    if observe is not None and ys is None:
        raise ValueError("observe needs the observed data ys, shape (len(ts), number of observed quantities).")
    ys = xs if observe is None else np.asarray(ys, float).reshape(len(ts), -1)
    if sigma is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # columns without data
            sigma = np.nanstd(ys, axis=0)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
    sigma = np.broadcast_to(np.asarray(sigma, float), ys.shape[1:])

    seeds, ode_points = _points(ts, n_seeds, seeds, n_ode_points, ode_points)
    D, p = xs.shape[1], len(theta_0)
    problem = _Problem(f, ts, seeds, ode_points, ys, sigma, observe, D, p, power)
    evaluate, linearize = _cached(problem.evaluate), _cached(problem.linearize)

    theta_0 = _clip(theta_0, bounds)
    U_0 = _initial_states(f, ts, xs, seeds, theta_0, unobserved_inits)
    result = _minimize(evaluate, linearize, U_0, theta_0, bounds, kwargs)

    U, theta = problem.unpack(result.x)
    result.theta, result.x, result.x_ts = theta, U, problem.V_data @ U
    result.seeds, result.ode_points, result.sigma, result.cost = seeds, ode_points, sigma, result.fun
    return result
