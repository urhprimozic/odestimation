"""
One estimator for x' = f(x, t, θ) that treats every component in one of two ways:

- matched: the component is an interpolant x̂ of its data (e.g. a GP), and its equation a residual
      derivative form:  x̂'(t_i) - f(x(t_i), t_i, θ)                     at the matching points t_i
      integral form:    x̂(b_j) - x̂(a_j) - ∫_{a_j}^{b_j} f(x(s), s, θ) ds   over windows [a_j, b_j]
  (gradient and integral matching). An interpolant is not an exact solution, so it can't be a constraint.
- free: the component's values U at the seeds are unknowns, linear in between, and its equation a
  constraint, one per gap between seeds: x_{k+1} - x_k = ∫ f over the gap, the integral the mean of f
  at the gap's ODE points times the gap (collocation; with the midpoint the implicit midpoint rule).
  Free components with data get a data term (x(t_j) - y_j) / σ; outputs computed from the states
  (`observe`) get one in any case.

Both kinds enter f together, x = (x̂_matched, x_free). Every residual is linear in values of f - an
identity at the matching points, or a quadrature over windows - so both forms share the code
(`operators`). Minimizes ½ Σ |r_i|^power over the scaled residuals subject to the constraint: with
free components by trust-constr, else by least_squares.
"""

import functools
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, NonlinearConstraint, least_squares, minimize

from odestimate.gp.regressor import GP, GP1dim
from odestimate.gradient_matching import powered
from odestimate.matching.operators import (block_diagonal, gap, integral_windows, linear_interpolation, ode_rule,
                                           ode_windows, pointwise_derivatives)


def _cached(fn):
    """fn(z), remembered for the last few z: trust-constr asks for the loss, the constraint and
    their derivatives in separate calls at the same z."""
    cache = functools.lru_cache(maxsize=8)(lambda key: fn(np.frombuffer(key)))
    return lambda z: cache(np.asarray(z, float).tobytes())


def _clip(theta, bounds):
    if bounds is None:
        return theta
    return np.clip(theta, *(np.broadcast_to(np.asarray(b, float), theta.shape) for b in bounds))


def _apply(A, M, width):
    """A @ M over points, for M with `width` rows per point: kron(A, I_width) @ M; M itself if A is None."""
    return M if A is None else sparse.kron(A, sparse.identity(width, format="csr")) @ M


class _Problem:
    """
    Residuals r = (matching, data) and constraint g in z = (U.ravel(), θ), U (n_seeds, n_free) the free
    components at the seeds, row-major. At the matching nodes, the ODE points and the data times the
    matched components are the interpolant's mean, the free ones V @ U.
    """

    def __init__(self, f, D, p, matched, free, seeds, mean, power, match=None, ode=None, data=None, observe=None):
        self.f, self.observe, self.power = f, observe, power
        self.M, self.H, self.D, self.p = matched, free, D, p
        self.n = len(seeds)
        self.I_H = sparse.identity(len(free), format="csr")
        self.points = {}
        if match is not None:  # (nodes, A, target (rows, n_matched), s)
            self.points["match"], self.A_match, self.target, self.s = match
        if ode is not None:  # (ode points, S, A)
            self.points["ode"], self.S_ode, self.A_ode = ode
        if data is not None:  # (ts, ys (n, channels), sigma)
            self.points["data"], ys, self.sigma = data
            self.mask = np.isfinite(ys)
            self.ys = np.where(self.mask, ys, 0.0)
        self.mean = {where: mean(t) for where, t in self.points.items()}
        self.V = {where: linear_interpolation(t, seeds)[0] for where, t in self.points.items()} if free else {}

    def unpack(self, z):
        return z[: self.n * len(self.H)].reshape(self.n, len(self.H)), z[self.n * len(self.H) :]

    def states(self, U, where):
        """Every component (m, D) at the matching nodes, the ODE points or the data times."""
        X = np.empty((len(self.points[where]), self.D))
        X[:, self.M] = self.mean[where]
        if self.H:
            X[:, self.H] = self.V[where] @ U
        return X

    def _f(self, X, where, theta):
        return np.asarray(self.f(X, self.points[where], theta), float).reshape(len(X), self.D)

    def _outputs(self, X, theta):
        """What the data measures: `observe`, else the free components."""
        if self.observe is None:
            return X[:, self.H]
        return np.asarray(self.observe(X, self.points["data"], theta), float).reshape(len(X), -1)

    def _has_data(self):
        return "data" in self.points and self.mask.any()

    def _residuals(self, U, theta):
        r = []
        if self.M:
            F = self._f(self.states(U, "match"), "match", theta)[:, self.M]
            r.append(((self.target - (F if self.A_match is None else self.A_match @ F)) / self.s).ravel())
        if self._has_data():
            r.append(((self._outputs(self.states(U, "data"), theta) - self.ys) / self.sigma)[self.mask])
        return np.concatenate(r) if r else np.zeros(0)

    def evaluate(self, z):
        """(r, g): the residuals (powered, see `powered`) and the constraint."""
        U, theta = self.unpack(z)
        g = np.zeros(0)
        if self.H:
            F = self._f(self.states(U, "ode"), "ode", theta)[:, self.H]
            g = (self.S_ode @ U - (F if self.A_ode is None else self.A_ode @ F)).ravel()
        return powered(self._residuals(U, theta), self.power), g

    def _by_free(self, derivatives, where):
        """d(rows)/dU from pointwise derivatives (m, rows per point, D) with respect to the states."""
        return block_diagonal(np.ascontiguousarray(derivatives[:, :, self.H])) @ sparse.kron(self.V[where], self.I_H)

    def _jacobian(self, dU, dtheta):
        """[d/dU, d/dθ], the first left out without free components."""
        return sparse.hstack([dU, dtheta] if self.H else [dtheta], format="csr")

    def linearize(self, z):
        """(r, J_r, g, J_g), the Jacobians sparse; derivatives of f and of the observation by forward
        differences."""
        U, theta = self.unpack(z)
        J_r = []
        if self.M:
            X = self.states(U, "match")
            Fx, Ft = pointwise_derivatives(self.f, X, self.points["match"], theta, self._f(X, "match", theta))
            scale = 1 / self.s[None, :, None]
            dU = self._by_free(-Fx[:, self.M] * scale, "match") if self.H else None
            dtheta = sparse.csr_matrix((-Ft[:, self.M] * scale).reshape(-1, self.p))
            if self.A_match is not None:
                dU = _apply(self.A_match, dU, len(self.M)) if self.H else None
                dtheta = _apply(self.A_match, dtheta, len(self.M))
            J_r.append(self._jacobian(dU, dtheta))
        if self._has_data():
            X = self.states(U, "data")
            if self.observe is None:
                Gx = np.zeros((len(X), len(self.H), self.D))
                Gx[:, np.arange(len(self.H)), self.H] = 1.0
                Gt = np.zeros((len(X), len(self.H), self.p))
            else:
                Gx, Gt = pointwise_derivatives(self.observe, X, self.points["data"], theta, self._outputs(X, theta))
            scale = 1 / self.sigma[None, :, None]
            dU = self._by_free(Gx * scale, "data") if self.H else None
            J_r.append(self._jacobian(dU, sparse.csr_matrix((Gt * scale).reshape(-1, self.p)))[self.mask.ravel()])
        J_r = J_r[0] if len(J_r) == 1 else sparse.vstack(J_r, format="csr")
        r = self._residuals(U, theta)
        if self.power != 2:  # ρ = sign(r) |r|^(power/2), dρ/dr = power/2 |r|^(power/2 - 1)
            J_r = sparse.diags(self.power / 2 * np.abs(r) ** (self.power / 2 - 1)) @ J_r
            r = powered(r, self.power)

        g, J_g = np.zeros(0), sparse.csr_matrix((0, len(z)))
        if self.H:
            X = self.states(U, "ode")
            F = self._f(X, "ode", theta)
            Fx, Ft = pointwise_derivatives(self.f, X, self.points["ode"], theta, F)
            dU = _apply(self.A_ode, self._by_free(Fx[:, self.H], "ode"), len(self.H))
            dtheta = _apply(self.A_ode, sparse.csr_matrix(-Ft[:, self.H].reshape(-1, self.p)), len(self.H))
            J_g = sparse.hstack([sparse.kron(self.S_ode, self.I_H) - dU, dtheta], format="csr")
            F = F[:, self.H]
            g = (self.S_ode @ U - (F if self.A_ode is None else self.A_ode @ F)).ravel()
        return r, J_r, g, J_g


def _points(ts, n_seeds, seeds, n_ode_points, ode_points, n_ode_per_gap):
    """Seeds and ODE points from the arguments of `matching`, see there."""
    if seeds is None:
        seeds = ts if n_seeds is None else np.linspace(ts[0], ts[-1], n_seeds)
    seeds = np.asarray(seeds, float)
    if ode_points is None and n_ode_points is None:
        if n_ode_per_gap == 1:
            ode_points = (seeds[:-1] + seeds[1:]) / 2
        else:
            parts = (np.arange(n_ode_per_gap) + 0.5) / n_ode_per_gap
            ode_points = (seeds[:-1, None] + np.diff(seeds)[:, None] * parts).ravel()
    elif ode_points is None:
        width = (seeds[-1] - seeds[0]) / n_ode_points
        ode_points = seeds[0] + width * (np.arange(n_ode_points) + 0.5)
    ode_points = np.asarray(ode_points, float)

    empty = np.sum(np.bincount(gap(ode_points, seeds), minlength=len(seeds) - 1) == 0)
    if empty:
        warnings.warn(f"{empty} gaps between seeds have no ODE point: there the free states follow the data, not the ODE.")
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
    result.cost = result.fun
    return result


def _least_squares(problem, theta_0, bounds, kwargs):
    """Minimizes ½ |r(θ)|² over θ within `bounds` (no free components), with least_squares."""
    bounds = (-np.inf, np.inf) if bounds is None else bounds
    return least_squares(lambda theta: problem.evaluate(theta)[0], theta_0,
                         jac=lambda theta: problem.linearize(theta)[1].toarray(), bounds=bounds, **kwargs)


def _channel_std(y):
    """Standard deviation of every column, NaN left out; 1 where it is 0 or there is no data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # columns without data
        s = np.nanstd(y, axis=0)
    return np.where(np.isfinite(s) & (s > 0), s, 1.0)


def matching(f, ts, xs, theta_0, matched="observed", form="derivative", gp=None, points=None, n_points=None,
             window=None, n_quad=10, unobserved=None, unobserved_inits=None, n_seeds=None, seeds=None,
             n_ode_points=None, ode_points=None, n_ode_per_gap=1, ode_window=None, observe=None, ys=None, sigma=None,
             s=None, bounds=None, power=2.0, gp_kwargs=None, **kwargs):
    """
    θ-parameter estimation for x' = f(x, t, θ) with matched and free components (module docstring).
    The named methods in `odestimate.matching` are this with `matched` and `form` set.

    Parameters
    ----------
    f : callable
        f(x, t, θ) -> dx/dt, numpy, batched: x (m, D), t (m,) -> (m, D). Must act pointwise (row i
        of the output depends only on row i of x).
    ts : array-like, shape (n,)
        Data times, increasing.
    xs : array-like, shape (n, D)
        Data on the states, NaN where missing; a column without data is an unobserved component.
        With `observe`, the data of the outputs is `ys` and xs only gives starting values.
    theta_0 : array-like, shape (p,)
        Initial guess for θ.
    matched : "observed", "all", "none" or list of int
        The matched components: the ones with data, all, none, or these columns. The rest are free.
    form : "derivative" or "integral"
        The matching residual: gradient or integral matching.
    gp : optional
        Interpolant of the matched components (in column order), fitted elsewhere: anything with
        `mean(t)` -> (m, n_matched), and `derivative(t)` for form="derivative", e.g. an
        `odestimate.gp.regressor.GP`. Default: one `GP1dim` per matched component on its data.
        Matching is only as good as the interpolant (and its derivative): check it.
    points, n_points :
        Matching points, or that many evenly spaced over ts. Default: ts.
    window : None, "cumulative" or float
        form="integral": the windows, see `operators.integral_windows`. Default: between consecutive
        matching points.
    n_quad : int
        form="integral": trapezoid steps between consecutive matching points.
    unobserved : iterable of int, optional
        Columns of xs to treat as unobserved.
    unobserved_inits : array-like, optional
        Start for the free components without data: shape (n_unobserved,), their state at ts[0],
        carried over the seeds with θ_0 by explicit Euler; or (len(ts), n_unobserved), a trajectory on
        ts. Default: state 0 at ts[0].
    n_seeds, seeds :
        Seeds of the free components: that many evenly spaced over ts, or these. Default: ts.
    n_ode_points, ode_points, n_ode_per_gap :
        ODE points of the free components: that many at the centres of equal parts of the seeds'
        range, or these, or n_ode_per_gap at the centres of equal parts of every gap (default 1: the
        midpoints). A gap's constraint uses the mean of f at its ODE points.
    ode_window : None, "cumulative" or float
        The free components' constraint: None, one per gap, x_{k+1} - x_k = ∫ f over the gap; or in
        integral form over windows of whole gaps ending at every seed, x(b) - x(a) = ∫_a^b f -
        "cumulative" from the first seed (x(t) = x(t_0) + ∫ f), or a length (`operators.ode_windows`).
        Exactly satisfied they describe the same trajectories (up to the quadrature); they differ in
        what the optimizer sees on the way.
    observe : callable, optional
        observe(x, t, θ) -> (m, q): an output measured at ts, computed from the states, batched and
        pointwise like f; its data is `ys`.
    ys : array-like, shape (n, q), optional
        The data for `observe`, NaN where missing.
    sigma : array-like, optional
        Noise std per data channel (the free components' data, or the outputs'). Default: the std of
        each channel's data.
    s : array-like, shape (n_matched,), optional
        Scale of the matching residual per matched component. Default: the std of its target (the
        interpolant's derivative or increments). With a data term too, only the ratio of s and sigma
        matters.
    bounds : (lo, hi), optional
        Bounds on θ.
    power : float
        Exponent of the loss, ½ Σ |r_i|^power over the scaled residuals: 2 is least squares, a larger
        one punishes large deviations more.
    gp_kwargs : dict, optional
        Forwarded to every `GP1dim` when the interpolant is fitted here. Default: normalize_y=True,
        n_restarts_optimizer=5.
    **kwargs :
        With free components to `scipy.optimize.minimize`, over the defaults method="trust-constr",
        the Gauss-Newton gradient and Hessian and the constraint; `options` are merged over gtol =
        barrier_tol = 1e-4. Without, to `scipy.optimize.least_squares` (ftol, xtol, gtol, max_nfev).

    Returns
    -------
    scipy.optimize.OptimizeResult
        The optimizer's result plus .theta, .x (n_seeds, n_free) the free components at .seeds,
        .x_ts (n, D) every component at ts (the matched ones the interpolant's mean), .gp, .points
        (the matching nodes), .ode_points, .matched, .free, .s, .sigma and .cost (the loss).
    """
    ts, xs, theta_0 = np.asarray(ts, float), np.array(xs, float), np.asarray(theta_0, float)
    if unobserved is not None:
        xs[:, list(unobserved)] = np.nan
    D, p = xs.shape[1], len(theta_0)
    with_data = [d for d in range(D) if np.isfinite(xs[:, d]).any()]
    M = {"observed": with_data, "all": list(range(D)), "none": []}[matched] if isinstance(matched, str) else sorted(matched)
    H = [d for d in range(D) if d not in M]
    if observe is not None and ys is None:
        raise ValueError("observe needs the observed data ys, shape (len(ts), number of observed quantities).")

    match = None
    if M:
        if gp is None:
            missing = [d for d in M if d not in with_data]
            if missing:
                raise ValueError(f"components {missing} are matched but have no data: pass an interpolant, gp=.")
            gp_kwargs = {"normalize_y": True, "n_restarts_optimizer": 5} if gp_kwargs is None else gp_kwargs
            gp = GP._from_gps([GP1dim(ts[np.isfinite(xs[:, d])], xs[np.isfinite(xs[:, d]), d], **gp_kwargs) for d in M])
        if np.shape(gp.mean(ts[:1]))[-1] != len(M):
            raise ValueError(f"the interpolant gp has {np.shape(gp.mean(ts[:1]))[-1]} components, {len(M)} are matched.")
        if points is None:
            points = ts if n_points is None else np.linspace(ts[0], ts[-1], n_points)
        points = np.asarray(points, float)
        if form == "derivative":
            nodes, A, target = points, None, gp.derivative(points)
        elif form == "integral":
            nodes, A, start, end = integral_windows(points, window, n_quad)
            at_nodes = gp.mean(nodes)
            target = at_nodes[end] - at_nodes[start]
        else:
            raise ValueError(f"form must be 'derivative' or 'integral', got {form!r}.")
        if s is None:
            s = np.std(target, axis=0)
            s = np.where(s > 0, s, 1.0)
        match = (nodes, A, target, np.broadcast_to(np.asarray(s, float), (len(M),)))
    mean = (lambda t: gp.mean(t)) if M else (lambda t: np.zeros((len(t), 0)))

    # C order: picking columns gives Fortran order, and a reduction over it sums in another order
    ys = np.ascontiguousarray(xs[:, H]) if observe is None else np.asarray(ys, float).reshape(len(ts), -1)
    sigma = _channel_std(ys) if sigma is None else sigma
    sigma = np.broadcast_to(np.asarray(sigma, float), ys.shape[1:])

    ode, seeds_ = None, np.zeros(0)
    if H:
        seeds_, ode_points = _points(ts, n_seeds, seeds, n_ode_points, ode_points, n_ode_per_gap)
        S, A = ode_rule(ode_points, seeds_) if ode_window is None else ode_windows(ode_points, seeds_, ode_window)
        ode = (ode_points, S, A)
    problem = _Problem(f, D, p, M, H, seeds_, mean, power, match=match, ode=ode, data=(ts, ys, sigma),
                       observe=observe)

    theta_0 = _clip(theta_0, bounds)
    if H:
        xs_start = xs.copy()
        xs_start[:, M] = mean(ts)
        U_0 = np.ascontiguousarray(_initial_states(f, ts, xs_start, seeds_, theta_0, unobserved_inits)[:, H])
        result = _minimize(_cached(problem.evaluate), _cached(problem.linearize), U_0, theta_0, bounds, kwargs)
    else:
        result = _least_squares(problem, theta_0, bounds, kwargs)

    U, theta = problem.unpack(result.x)
    result.theta, result.x, result.x_ts, result.gp = theta, U, problem.states(U, "data"), gp
    result.points = None if match is None else match[0]
    result.seeds, result.ode_points = seeds_, None if ode is None else ode[0]
    result.matched, result.free, result.s, result.sigma = M, H, None if match is None else match[3], sigma
    return result
