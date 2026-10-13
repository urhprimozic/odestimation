"""
Gradient matching for the observed components, collocation for the unobserved ones. For
x' = f(x, t, θ) with observed components o and unobserved components h:

    argmin_{U_h, θ}  ½ Σ_i Σ_o ((x̂_o'(t_i) - f_o(x(t_i), t_i, θ)) / s_o)²   [+ ½ Σ_j Σ_c ((g_c(x(t_j), t_j, θ) - y_jc) / σ_c)²]
    subject to       x_h'(e) - f_h(x(e), e, θ) = 0   at the ODE points e,

with x = (x̂_o, x_h):
- observed state: a GP of its data, x̂_o its mean and x̂_o' its derivative. Its equation is a
  residual, as in gradient matching - a GP is not an exact solution, so it can't be a constraint.
- unobserved state: linear between seeds, its values there U_h unknowns, as in `collocation`. Its
  equation is a constraint, so x_h is a solution of the model driven by x̂_o, its initial state free.
- observed output computed from the states (`observe`, e.g. a flow from the water stored in
  unobserved states): a data term, as in `collocation`.
- exogenous variables: part of f.
Without unobserved components this is gradient matching; with every component unobserved and an
`observe`, it is `collocation`.
"""

import warnings

import numpy as np
from scipy import sparse

from odestimate.gp.regressor import GP, GP1dim
from odestimate.gradient_matching import powered
from odestimate.latent.collocation import (_block_diagonal, _cached, _clip, _initial_states, _linear_interpolation,
                                           _minimize, _points)
from odestimate.latent.magi import _pointwise_derivatives


class _Problem:
    """
    Residuals r = (gradient matching, data) and constraint g in z = (U_h.ravel(), θ), U_h (n_seeds, D_h)
    the unobserved components at the seeds, row-major as in `collocation`. The observed components
    at the gradient matching points, the ODE points and the data times are the GP's mean there.
    """

    def __init__(self, f, observe, observed, hidden, D, p, gm_points, dx_gm, s, ode_points, data_times, ys, sigma,
                 seeds, gp_mean, power):
        self.f, self.observe, self.power = f, observe, power
        self.obs, self.hid, self.D, self.p = observed, hidden, D, p
        self.n, self.D_h = len(seeds), len(hidden)
        self.points = {"gm": gm_points, "ode": ode_points, "data": data_times}
        self.mean = {where: gp_mean(t) for where, t in self.points.items()}  # observed components there
        self.V = {where: _linear_interpolation(t, seeds)[0] for where, t in self.points.items()}
        _, self.S_ode = _linear_interpolation(ode_points, seeds)
        self.dx_gm, self.s = dx_gm, s
        self.mask = np.isfinite(ys) if observe is not None else np.zeros((len(data_times), 0), bool)
        self.ys = np.where(self.mask, ys, 0.0) if observe is not None else None
        self.sigma = sigma
        self.I_h = sparse.identity(self.D_h, format="csr")

    def unpack(self, z):
        return z[: self.n * self.D_h].reshape(self.n, self.D_h), z[self.n * self.D_h :]

    def states(self, U_h, where):
        """The full states (m, D) at the gradient matching points, the ODE points or the data times."""
        X = np.empty((len(self.points[where]), self.D))
        X[:, self.obs] = self.mean[where]
        X[:, self.hid] = self.V[where] @ U_h
        return X

    def _f(self, X, where, theta):
        return np.asarray(self.f(X, self.points[where], theta), float).reshape(len(X), self.D)

    def _observation(self, X, theta):
        return np.asarray(self.observe(X, self.points["data"], theta), float).reshape(len(X), -1)

    def _residuals(self, U_h, theta):
        X = self.states(U_h, "gm")
        r = [((self.dx_gm - self._f(X, "gm", theta)[:, self.obs]) / self.s).ravel()]
        if self.observe is not None:
            r.append(((self._observation(self.states(U_h, "data"), theta) - self.ys) / self.sigma)[self.mask])
        return np.concatenate(r)

    def evaluate(self, z):
        """(r, g): the residuals (powered, see `powered`) and g at every ODE point and unobserved component."""
        U_h, theta = self.unpack(z)
        g = (self.S_ode @ U_h - self._f(self.states(U_h, "ode"), "ode", theta)[:, self.hid]).ravel()
        return powered(self._residuals(U_h, theta), self.power), g

    def _by_hidden(self, derivatives, where):
        """d(rows)/dU_h from pointwise derivatives (m, rows per point, D) with respect to the states."""
        return _block_diagonal(np.ascontiguousarray(derivatives[:, :, self.hid])) @ sparse.kron(self.V[where], self.I_h)

    def linearize(self, z):
        """(r, J_r, g, J_g), the Jacobians sparse; derivatives of f and of the observation by forward
        differences."""
        U_h, theta = self.unpack(z)

        X = self.states(U_h, "gm")
        Fx, Ft = _pointwise_derivatives(self.f, X, self.points["gm"], theta, self._f(X, "gm", theta))
        scale = 1 / self.s[None, :, None]
        J_r = [sparse.hstack([self._by_hidden(-Fx[:, self.obs] * scale, "gm"),
                              sparse.csr_matrix((-Ft[:, self.obs] * scale).reshape(-1, self.p))], format="csr")]
        if self.observe is not None:
            X = self.states(U_h, "data")
            Gx, Gt = _pointwise_derivatives(self.observe, X, self.points["data"], theta, self._observation(X, theta))
            scale = 1 / self.sigma[None, :, None]
            J_r.append(sparse.hstack([self._by_hidden(Gx * scale, "data"),
                                      sparse.csr_matrix((Gt * scale).reshape(-1, self.p))], format="csr")[self.mask.ravel()])
        J_r = sparse.vstack(J_r, format="csr")
        r = self._residuals(U_h, theta)
        if self.power != 2:  # ρ = sign(r) |r|^(power/2), dρ/dr = power/2 |r|^(power/2 - 1)
            J_r = sparse.diags(self.power / 2 * np.abs(r) ** (self.power / 2 - 1)) @ J_r
            r = powered(r, self.power)

        X = self.states(U_h, "ode")
        F = self._f(X, "ode", theta)
        Fx, Ft = _pointwise_derivatives(self.f, X, self.points["ode"], theta, F)
        J_g = sparse.hstack([sparse.kron(self.S_ode, self.I_h) - self._by_hidden(Fx[:, self.hid], "ode"),
                             sparse.csr_matrix(-Ft[:, self.hid].reshape(-1, self.p))], format="csr")
        g = (self.S_ode @ U_h - F[:, self.hid]).ravel()
        return r, J_r, g, J_g


def gm_collocation(f, ts, xs, theta_0, gp=None, unobserved=None, unobserved_inits=None, n_ts=None, gm_points=None,
                   n_seeds=None, seeds=None, n_ode_points=None, ode_points=None, observe=None, ys=None, sigma=None,
                   s=None, bounds=None, power=2.0, gp_kwargs=None, **kwargs):
    """
    θ-parameter estimation for x' = f(x, t, θ): gradient matching on the observed components,
    collocation for the unobserved ones (see the module docstring), with trust-constr and a
    Gauss-Newton model of the loss.

    Parameters
    ----------
    f : callable
        f(x, t, θ) -> dx/dt, numpy, batched: x (m, D), t (m,) -> (m, D). Must act pointwise (row i
        of the output depends only on row i of x).
    ts : array-like, shape (n,)
        Data times, increasing.
    xs : array-like, shape (n, D)
        Data on the states, NaN where missing; a column without data is an unobserved component.
    theta_0 : array-like, shape (p,)
        Initial guess for θ.
    gp : odestimate.gp.regressor.GP, optional
        A GP of the observed components (in column order), fitted elsewhere: its `mean(t)` and
        `derivative(t)`, (m, n_observed), are used instead of fitting one. Default: one `GP1dim` per
        observed component on its data (`gp_kwargs`). Gradient matching is only as good as the GP's
        derivative: check it (on FitzHugh-Nagumo, sklearn's defaults give a length scale of 0.04
        and a derivative off by 640 %).
    unobserved : iterable of int, optional
        Columns of xs to treat as unobserved.
    unobserved_inits : array-like, optional
        Start for the unobserved components: shape (n_unobserved,), their state at ts[0], carried
        over the seeds with θ_0 by explicit Euler (the observed ones from the GP); or
        (len(ts), n_unobserved), a trajectory on ts. Default: state 0 at ts[0].
    n_ts : int, optional
        Number of gradient matching points, evenly spaced over ts.
    gm_points : array-like, optional
        The gradient matching points themselves (instead of n_ts). Default: ts.
    n_seeds, seeds, n_ode_points, ode_points :
        Seeds and ODE points of the unobserved components, see `collocation`. Default: seeds ts,
        ODE points their midpoints.
    observe : callable, optional
        observe(x, t, θ) -> (m, q): an output measured at ts, computed from the states, batched and
        pointwise like f; its data is `ys`.
    ys : array-like, shape (n, q), optional
        The data for `observe`, NaN where missing.
    sigma : array-like, shape (q,), optional
        Noise std per channel of ys. Default: the std of each channel's data.
    s : array-like, shape (n_observed,), optional
        Scale of the gradient matching residual per observed component. Default: the std of the
        GP's derivative over the gradient matching points. With both terms, only the ratio of s and
        sigma matters - it weighs the derivatives against the outputs.
    bounds : (lo, hi), optional
        Bounds on θ.
    power : float
        Exponent of the loss, ½ Σ |r_i|^power over the scaled residuals of both terms: 2 is least
        squares, a larger one punishes large deviations more.
    gp_kwargs : dict, optional
        Forwarded to every `GP1dim` when the GP is fitted here. Default: normalize_y=True,
        n_restarts_optimizer=5.
    **kwargs :
        Forwarded to `scipy.optimize.minimize`, as in `collocation`.

    Returns
    -------
    scipy.optimize.OptimizeResult
        minimize's result plus .theta, .x (n_seeds, n_unobserved) the unobserved components at
        .seeds, .x_ts (n, D) every component at ts (the observed ones the GP's mean), .gp, .gm_points,
        .ode_points, .observed, .unobserved, .s, .sigma and .cost (the loss at the optimum).
    """
    ts, xs, theta_0 = np.asarray(ts, float), np.array(xs, float), np.asarray(theta_0, float)
    if unobserved is not None:
        xs[:, list(unobserved)] = np.nan
    D, p = xs.shape[1], len(theta_0)
    observed = [d for d in range(D) if np.isfinite(xs[:, d]).any()]
    hidden = [d for d in range(D) if d not in observed]
    if observe is not None and ys is None:
        raise ValueError("observe needs the observed data ys, shape (len(ts), number of observed quantities).")

    if gp is None:
        gp_kwargs = {"normalize_y": True, "n_restarts_optimizer": 5} if gp_kwargs is None else gp_kwargs
        gp = GP._from_gps([GP1dim(ts[np.isfinite(xs[:, d])], xs[np.isfinite(xs[:, d]), d], **gp_kwargs) for d in observed])
    if len(gp) != len(observed):
        raise ValueError(f"gp has {len(gp)} components, the data {len(observed)} observed ones.")
    gp_mean = (lambda t: gp.mean(t)) if observed else (lambda t: np.zeros((len(t), 0)))

    if gm_points is None:
        gm_points = ts if n_ts is None else np.linspace(ts[0], ts[-1], n_ts)
    gm_points = np.asarray(gm_points, float)
    dx_gm = gp.derivative(gm_points) if observed else np.zeros((len(gm_points), 0))
    if s is None:
        s = np.std(dx_gm, axis=0)
        s = np.where(s > 0, s, 1.0)
    s = np.broadcast_to(np.asarray(s, float), (len(observed),))
    if observe is not None:
        ys = np.asarray(ys, float).reshape(len(ts), -1)
        if sigma is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # channels without data
                sigma = np.nanstd(ys, axis=0)
            sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
        sigma = np.broadcast_to(np.asarray(sigma, float), ys.shape[1:])

    seeds, ode_points = _points(ts, n_seeds, seeds, n_ode_points, ode_points)
    problem = _Problem(f, observe, observed, hidden, D, p, gm_points, dx_gm, s, ode_points, ts, ys, sigma, seeds,
                       gp_mean, power)
    evaluate, linearize = _cached(problem.evaluate), _cached(problem.linearize)

    theta_0 = _clip(theta_0, bounds)
    xs_start = xs.copy()
    xs_start[:, observed] = gp_mean(ts)
    U_0 = _initial_states(f, ts, xs_start, seeds, theta_0, unobserved_inits)[:, hidden]
    result = _minimize(evaluate, linearize, U_0, theta_0, bounds, kwargs)

    U_h, theta = problem.unpack(result.x)
    x_ts = np.empty((len(ts), D))
    x_ts[:, observed], x_ts[:, hidden] = gp_mean(ts), problem.V["data"] @ U_h
    result.theta, result.x, result.x_ts, result.gp = theta, U_h, x_ts, gp
    result.seeds, result.gm_points, result.ode_points = seeds, gm_points, ode_points
    result.observed, result.unobserved, result.s, result.sigma, result.cost = observed, hidden, s, sigma, result.fun
    return result
