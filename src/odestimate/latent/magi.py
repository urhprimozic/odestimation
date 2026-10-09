"""
MAGI - MAnifold-constrained Gaussian process Inference (Yang, Wong, Kou, PNAS 2021): parameter
estimation for x' = f(x, t, θ) without numerical integration, components without data allowed.

Every component gets a GP prior; its values at a set of seeds S are unknowns, together with θ. The
ODE enters as the condition that the GP's derivative equals f. Derivation and notation: PyBM
`notes/magi.md` (section numbers below refer to it).

Seeds and loss points. In MAGI as in the paper the seeds are a grid I that contains the data times,
and the ODE is imposed on the same grid. With `n_seeds` (or a `grid` without all data times) the
seeds are fewer, the data and ODE terms are evaluated at other points (by default the data times),
and the states there are the GP's conditional mean given the seeds. That is the "inducing points"
(DTC) approximation of GP regression: it ignores the conditional variance of the states between
seeds. It keeps the number of unknowns at n_seeds * D + p, independent of the amount of data.

With `midpoints`, the seeds and the ODE points lie halfway between consecutive data times and only
the data term is evaluated at the data times. That is for inputs that are piecewise constant
between data times (daily totals with step interpolation): the solution's derivative jumps at the
data times, where a GP derivative averages the slopes on both sides while f sees only the new
input. Halfway between, the input is constant and the GP derivative is close to the secant.

method="max_likelihood": the posterior's mode (MAP) by Gauss-Newton - fast (§5).
method="bayes": samples from the posterior by Hamiltonian Monte Carlo, started at the MAP (§3.8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np
from scipy.linalg import LinAlgError, cho_solve, cholesky, solve_triangular
from scipy.optimize import OptimizeResult, least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

KernelKind = Literal["matern52", "matern32", "rbf"]


# ------------------------------------------------------------------------------------ kernels ----


def _kernel_derivatives(kind: KernelKind, signal_var: float, length_scale: float, r: np.ndarray):
    """kappa(r), kappa'(r) and -kappa''(r) of a stationary kernel k(s, t) = kappa(s - t), so that
    k = kappa, d/ds k = kappa' and d^2/(ds dt) k = -kappa'' (notes §6.6)."""
    s2, ell = signal_var, length_scale
    if kind == "rbf":
        k = s2 * np.exp(-0.5 * (r / ell) ** 2)
        return k, -k * r / ell**2, k * (1 / ell**2 - r**2 / ell**4)
    a = np.sqrt(3.0 if kind == "matern32" else 5.0) / ell
    u = a * np.abs(r)
    e = np.exp(-u)
    if kind == "matern32":
        return s2 * (1 + u) * e, -s2 * a**2 * r * e, s2 * a**2 * (1 - u) * e
    if kind == "matern52":
        return s2 * (1 + u + u**2 / 3) * e, -s2 * a**2 * r / 3 * (1 + u) * e, s2 * a**2 / 3 * (1 + u - u**2) * e
    raise ValueError(f"Unknown kernel {kind!r}, use 'matern52', 'matern32' or 'rbf'.")


def _sklearn_kernel(kind: KernelKind, length_scale: float, length_scale_bounds):
    if kind == "rbf":
        shape = RBF(length_scale, length_scale_bounds)
    else:
        shape = Matern(length_scale, length_scale_bounds, nu=1.5 if kind == "matern32" else 2.5)
    return ConstantKernel(1.0, (1e-3, 1e3)) * shape + WhiteKernel(1e-2, (1e-8, 1.0))


def _cholesky_jitter(A: np.ndarray, scale: float) -> np.ndarray:
    """Lower Cholesky factor of A, adding the smallest diagonal jitter (relative to `scale`) that
    makes it succeed: on dense grids C and Psi are numerically singular (notes §6.7)."""
    n = len(A)
    for jitter in (0.0, *10.0 ** np.arange(-12, -3)):
        try:
            return cholesky(A + jitter * scale * np.eye(n), lower=True)
        except LinAlgError:
            continue
    raise LinAlgError("Covariance matrix not positive definite even with jitter 1e-4; use fewer seeds or a rougher kernel.")


@dataclass
class _Component:
    """
    The GP of one component, given its values u at the seeds S (notes §2.2):
        prior        L_C^-1 (u - μ),                     C = k(S, S)
        state at E   μ + Phi (u - μ),                    Phi = k(E, S) C^-1   (None: E = S, identity)
        ODE          L_Psi^-1 f(x(E)) - A (u - μ),       A = L_Psi^-1 'k(E, S) C^-1,
                     Psi = k''(E, E) - 'k(E, S) C^-1 k'(S, E), the derivative's covariance given u
        data         μ + Phi_data (u - μ) at the data times
    """

    mu: float
    LiC: np.ndarray
    LiP: np.ndarray
    A: np.ndarray
    Phi: "np.ndarray | None"
    Phi_data: np.ndarray


def _component(kind: KernelKind, signal_var: float, length_scale: float, mu: float, seeds, ode_points,
               data_times) -> _Component:
    k_ss, _, _ = _kernel_derivatives(kind, signal_var, length_scale, seeds[:, None] - seeds[None, :])
    LC = _cholesky_jitter(k_ss, signal_var)
    solve = lambda B: cho_solve((LC, True), B.T).T  # B C^-1
    same = ode_points is seeds
    k_es, dk_es, _ = _kernel_derivatives(kind, signal_var, length_scale, ode_points[:, None] - seeds[None, :])
    _, _, ddk_ee = _kernel_derivatives(kind, signal_var, length_scale, ode_points[:, None] - ode_points[None, :])
    m = solve(dk_es)  # 'k(E, S) C^-1
    psi = ddk_ee - m @ dk_es.T  # k'(S, E) = 'k(E, S)^T
    LP = _cholesky_jitter(0.5 * (psi + psi.T), ddk_ee[0, 0])  # ddk[0, 0] = Var(x')
    LiP = solve_triangular(LP, np.eye(len(ode_points)), lower=True)
    position = np.clip(np.searchsorted(seeds, data_times), 0, len(seeds) - 1)
    if np.allclose(seeds[position], data_times):  # data times are seeds: exact selection, not k C^-1
        Phi_data = np.eye(len(seeds))[position]
    else:
        k_ds, _, _ = _kernel_derivatives(kind, signal_var, length_scale, data_times[:, None] - seeds[None, :])
        Phi_data = solve(k_ds)
    return _Component(
        mu=mu,
        LiC=solve_triangular(LC, np.eye(len(seeds)), lower=True),
        LiP=LiP,
        A=LiP @ m,
        Phi=None if same else solve(k_es),
        Phi_data=Phi_data,
    )


def _pointwise_derivatives(g, X, t, theta, G):
    """dG/dx (m, q, D) and dG/dθ (m, q, p) of a pointwise function G = g(X, t, θ), by forward
    differences: perturbing one column of every row at once perturbs each point independently,
    so it takes one call of g per state dimension and per parameter."""
    eps = np.sqrt(np.finfo(float).eps)
    m, D = X.shape
    Gx = np.empty((m, G.shape[1], D))
    for e in range(D):
        h = eps * np.maximum(1.0, np.abs(X[:, e]))
        Xp = X.copy()
        Xp[:, e] += h
        Gx[:, :, e] = (np.asarray(g(Xp, t, theta), float).reshape(G.shape) - G) / h[:, None]
    Gt = np.empty((m, G.shape[1], len(theta)))
    for k in range(len(theta)):
        h = eps * max(1.0, abs(theta[k]))
        tp = theta.copy()
        tp[k] += h
        Gt[:, :, k] = (np.asarray(g(X, t, tp), float).reshape(G.shape) - G) / h
    return Gx, Gt


# ------------------------------------------------------------------------------------ problem ----


class _Problem:
    """
    The negative log posterior (notes (3.2)) as residuals, z = (u_1, ..., u_D, θ), u_d the values of
    component d at the seeds:
        data:  (g(x(t_i), t_i, θ) - y_i) / σ_c   per data channel c (g = the state itself unless
                                                  `observe` is given), x(t_i) interpolated from u
        prior: β^-1/2 L_C^-1 (u_d - μ_d)
        ODE:   β^-1/2 (L_Psi^-1 f_d(x(E), θ, E) - A_d (u_d - μ_d)),  E the ODE points
    (the prior mean is constant, so its derivative is 0).
    """

    def __init__(self, f, jac, seeds, ode_points, data_times, components, beta, p, ys, observe):
        self.f, self.jac, self.observe = f, jac, observe
        self.seeds, self.ode_points, self.data_times, self.comp = seeds, ode_points, data_times, components
        self.n, self.D, self.p = len(seeds), len(components), p
        self.sqrt_beta = np.sqrt(beta)
        self.mask = np.isfinite(ys)  # (n_t, channels)
        self.ys = np.where(self.mask, ys, 0.0)
        self.counts = self.mask.sum(0)
        self.slices = np.split(np.arange(self.mask.sum()), np.cumsum(self.counts)[:-1])
        self.n_data = int(self.mask.sum())

    def unpack(self, z):
        return z[: self.n * self.D].reshape(self.D, self.n).T, z[self.n * self.D :]

    def states(self, U, where: Literal["ode", "data"]):
        """The states at the ODE points or the data times, from the seed values U (n, D)."""
        cols = []
        for d, c in enumerate(self.comp):
            Phi = c.Phi if where == "ode" else c.Phi_data
            cols.append(U[:, d] if Phi is None else c.mu + Phi @ (U[:, d] - c.mu))
        return np.stack(cols, axis=1)

    def _f(self, X, theta):
        return np.asarray(self.f(X, self.ode_points, theta), dtype=float).reshape(len(self.ode_points), self.D)

    def _g(self, X, theta):
        if self.observe is None:
            return X
        return np.asarray(self.observe(X, self.data_times, theta), float).reshape(len(self.data_times), -1)

    def data_residuals(self, z, sigma):
        U, theta = self.unpack(z)
        r = (self._g(self.states(U, "data"), theta) - self.ys) / sigma[None, :]
        return r.T[self.mask.T]  # channel by channel, only where there is data

    def data_jacobian(self, z, sigma):
        U, theta = self.unpack(z)
        X = self.states(U, "data")
        n_t, n_c = self.mask.shape
        J = np.zeros((self.n_data, self.n * self.D + self.p))
        if self.observe is None:
            Gx = np.broadcast_to(np.eye(self.D)[None], (n_t, n_c, self.D))
            Gt = np.zeros((n_t, n_c, self.p))
        else:
            Gx, Gt = _pointwise_derivatives(self.observe, X, self.data_times, theta, self._g(X, theta))
        for c, rows in enumerate(self.slices):
            at = np.flatnonzero(self.mask[:, c])
            for d, comp in enumerate(self.comp):
                J[rows, d * self.n : (d + 1) * self.n] = Gx[at, c, d][:, None] * comp.Phi_data[at] / sigma[c]
            J[rows, self.n * self.D :] = Gt[at, c, :] / sigma[c]
        return J

    def model_residuals(self, z):
        """Prior and ODE residuals."""
        U, theta = self.unpack(z)
        F = self._f(self.states(U, "ode"), theta)
        prior, ode = [], []
        for d, c in enumerate(self.comp):
            dev = U[:, d] - c.mu
            prior.append(c.LiC @ dev)
            ode.append(c.LiP @ F[:, d] - c.A @ dev)
        return np.concatenate(prior + ode) / self.sqrt_beta

    def model_jacobian(self, z):
        U, theta = self.unpack(z)
        X = self.states(U, "ode")
        F = self._f(X, theta)
        if self.jac is not None:
            Fx, Ft = (np.asarray(a, float) for a in self.jac(X, self.ode_points, theta))
        else:
            Fx, Ft = _pointwise_derivatives(self.f, X, self.ode_points, theta, F)
        n, D, n_e = self.n, self.D, len(self.ode_points)
        J = np.zeros((D * (n + n_e), n * D + self.p))
        r = np.empty(D * (n + n_e))
        for d, c in enumerate(self.comp):
            dev = U[:, d] - c.mu
            rows_prior = slice(d * n, (d + 1) * n)
            rows_ode = slice(D * n + d * n_e, D * n + (d + 1) * n_e)
            J[rows_prior, d * n : (d + 1) * n] = c.LiC
            for e, ce in enumerate(self.comp):
                # d f_d(x(E)) / d u_e = diag(df_d/dx_e) Phi_e
                block = c.LiP * Fx[:, d, e][None, :] if ce.Phi is None else c.LiP @ (Fx[:, d, e][:, None] * ce.Phi)
                J[rows_ode, e * n : (e + 1) * n] = block
            J[rows_ode, d * n : (d + 1) * n] -= c.A
            J[rows_ode, n * D :] = c.LiP @ Ft[:, d, :]
            r[rows_prior] = c.LiC @ dev
            r[rows_ode] = c.LiP @ F[:, d] - c.A @ dev
        return r / self.sqrt_beta, J / self.sqrt_beta


# ---------------------------------------------------------------------------- setup helpers ----


def refine_grid(ts, level: int = 1) -> np.ndarray:
    """ts with 2^level - 1 evenly spaced points inserted between neighbours (MAGI's
    `setDiscretization(level=...)`): a denser grid enforces the ODE at more points (notes §3.9)."""
    ts = np.asarray(ts, float)
    k = 2**level
    inner = ts[:-1, None] + np.arange(k)[None, :] * (np.diff(ts)[:, None] / k)
    return np.append(inner.ravel(), ts[-1])


def _fit_gp(kind, t, y, restarts, max_points=300):
    """GP fit of one series: (fitted regressor, signal variance, length scale, noise variance, mean),
    in the series' own units (MAGI step 1, notes §3.7). At most `max_points` evenly spaced points
    (the fit is O(n^3) per optimizer step). The length scale is kept above half the sampling step,
    below which the fit degenerates to interpolating the noise."""
    if len(t) > max_points:
        keep = np.linspace(0, len(t) - 1, max_points).round().astype(int)
        t, y = t[keep], y[keep]
    dt = np.diff(np.sort(t))
    bounds = (0.5 * dt[dt > 0].min(), 10 * (t.max() - t.min()))
    ell0 = float(np.clip(0.1 * (t.max() - t.min()), *bounds))
    gpr = GaussianProcessRegressor(_sklearn_kernel(kind, ell0, bounds), normalize_y=True,
                                   n_restarts_optimizer=restarts, random_state=0)
    gpr.fit(t[:, None], y)
    k = gpr.kernel_
    y_std = float(np.ravel(gpr._y_train_std)[0])
    return (gpr, k.k1.k1.constant_value * y_std**2, k.k1.k2.length_scale, k.k2.noise_level * y_std**2,
            float(np.ravel(gpr._y_train_mean)[0]))


def _setup(f, ts, xs, theta_0, unobserved, unobserved_inits, grid, n_seeds, ode_points, midpoints, kernel, phi,
           sigma, mu, beta, bounds, jac, gp_restarts, observe, ys):
    ts, xs, theta_0 = np.asarray(ts, float), np.array(xs, float), np.asarray(theta_0, float)
    D = xs.shape[1]
    if unobserved is not None:
        xs[:, list(unobserved)] = np.nan
    mid = (ts[:-1] + ts[1:]) / 2
    if n_seeds is not None and midpoints:
        seeds = mid[np.unique(np.linspace(0, len(mid) - 1, n_seeds).round().astype(int))]
    elif n_seeds is not None:
        seeds = np.linspace(ts[0], ts[-1], n_seeds)
    else:
        seeds = (mid if midpoints else ts) if grid is None else np.asarray(grid, float)
    if ode_points is None and midpoints:
        ode_points = mid
    elif ode_points is None:
        position = np.clip(np.searchsorted(seeds, ts), 0, len(seeds) - 1)
        ode_points = seeds if np.allclose(seeds[position], ts) else ts  # MAGI as in the paper, else the data times
    ode_points = np.asarray(ode_points, float)
    if len(ode_points) == len(seeds) and np.allclose(ode_points, seeds):
        ode_points = seeds  # the same points: identity instead of k C^-1 (checked with `is` below)
    if observe is not None and ys is None:
        raise ValueError("observe needs the observed data ys, shape (len(ts), number of observed quantities).")
    ys = xs if observe is None else np.asarray(ys, float).reshape(len(ts), -1)

    has_data = [np.isfinite(xs[:, d]) for d in range(D)]
    hidden = [d for d in range(D) if not has_data[d].any()]
    phi = [None] * D if phi is None else list(phi)
    phi_given = list(phi)
    mu = np.full(D, np.nan) if mu is None else np.array(mu, float)
    sigma = np.full(ys.shape[1], np.nan) if sigma is None else np.array(sigma, float)

    # starting trajectory of the hidden components, on the data times
    inits = np.zeros((len(ts), len(hidden)))
    if unobserved_inits is not None:
        given = np.asarray(unobserved_inits, float)
        inits[:] = given if given.ndim == 2 else np.broadcast_to(given, (len(ts), len(hidden)))
    inits_are_trajectory = unobserved_inits is not None and np.ndim(unobserved_inits) == 2

    # components with data on the state: GP prior and starting values from a GP fit (MAGI step 1)
    U0 = np.zeros((len(seeds), D))
    fitted_ell = []
    for d in range(D):
        if d in hidden:
            continue
        gpr, s2, ell, noise, mean = _fit_gp(kernel, ts[has_data[d]], xs[has_data[d], d], gp_restarts)
        fitted_ell.append(ell)
        phi[d] = (s2, ell) if phi[d] is None else phi[d]
        mu[d] = mean if np.isnan(mu[d]) else mu[d]
        if observe is None and np.isnan(sigma[d]):
            sigma[d] = np.sqrt(noise)
        U0[:, d] = gpr.predict(seeds[:, None])

    # Hidden components have no data. With a starting trajectory (e.g. a simulation with θ_0) the
    # prior is fitted to it; otherwise (heuristic, notes §6.5) the length scale of the observed
    # components, the starting value as mean and a signal std of the size of that mean. Pass phi
    # and mu for them whenever the scale is known better.
    for j, d in enumerate(hidden):
        if inits_are_trajectory and phi[d] is None:
            _, s2, ell, _, mean = _fit_gp(kernel, ts, inits[:, j], gp_restarts)
            phi[d] = (s2, ell)
            mu[d] = mean if np.isnan(mu[d]) else mu[d]
        if np.isnan(mu[d]):
            mu[d] = float(inits[:, j].mean())
        if phi[d] is None:
            ell = float(np.median(fitted_ell)) if fitted_ell else 0.1 * (ts[-1] - ts[0])
            phi[d] = (max(abs(mu[d]), 1.0) ** 2, ell)
        U0[:, d] = np.interp(seeds, ts, inits[:, j]) if unobserved_inits is not None else mu[d]

    # noise of every data channel: from the GP fit (direct observation: above; observed functions of
    # the state: a GP fit of each observed series)
    for c in range(ys.shape[1]):
        ok = np.isfinite(ys[:, c])
        if np.isnan(sigma[c]) and ok.sum() > 1:
            sigma[c] = np.sqrt(_fit_gp(kernel, ts[ok], ys[ok, c], gp_restarts)[3])

    # With seeds coarser than the data, a length scale below the seed spacing can't be represented
    # by the seeds: between them the GP falls back to its mean. Fitted (not given) length scales are
    # kept at least the spacing.
    if ode_points is not seeds:
        spacing = float(np.mean(np.diff(seeds)))
        phi = [(s2, max(ell, spacing)) if given is None else (s2, ell)
               for (s2, ell), given in zip(phi, phi_given)]
    components = [_component(kernel, *phi[d], mu[d], seeds, ode_points, ts) for d in range(D)]
    n_data = int(np.isfinite(ys).sum())
    if n_data == 0:
        raise ValueError("No data.")
    # notes §3.5: (number of GP terms / 2) / number of data points; D |I| / N for MAGI as in the paper
    beta = D * (len(seeds) + len(ode_points)) / (2 * n_data) if beta is None else float(beta)

    lo, hi = (-np.inf, np.inf) if bounds is None else bounds
    lo = np.broadcast_to(np.asarray(lo, float), theta_0.shape)
    hi = np.broadcast_to(np.asarray(hi, float), theta_0.shape)
    theta_0 = np.clip(theta_0, lo, hi)
    problem = _Problem(f, jac, seeds, ode_points, ts, components, beta, len(theta_0), ys, observe)
    z0 = np.concatenate([U0.T.ravel(), theta_0])
    z_lo = np.concatenate([np.full(len(seeds) * D, -np.inf), lo])
    z_hi = np.concatenate([np.full(len(seeds) * D, np.inf), hi])
    frozen_components = [d for d in range(D) if d not in hidden]
    return problem, z0, z_lo, z_hi, frozen_components, np.array(phi, float), sigma, mu, beta


# --------------------------------------------------------------------------- max likelihood ----


def _least_squares(problem, z0, free, sigma, z_lo, z_hi, kwargs):
    def full(z_free):
        z = z0.copy()
        z[free] = z_free
        return z

    def fun(z_free):
        z = full(z_free)
        return np.concatenate([problem.data_residuals(z, sigma), problem.model_residuals(z)])

    def jac(z_free):
        z = full(z_free)
        _, J = problem.model_jacobian(z)
        return np.vstack([problem.data_jacobian(z, sigma), J])[:, free]

    result = least_squares(fun, z0[free], jac=jac, bounds=(z_lo[free], z_hi[free]),
                           **{"method": "trf", "x_scale": "jac", **kwargs})
    return full(result.x), result


def _map(problem, z0, frozen_components, sigma, z_lo, z_hi, minimizer_kwargs):
    """
    MAP in two stages (notes §6.4). A: components with data on the state frozen at their GP
    posterior mean, fit θ and the rest - with every component observed this is gradient matching
    weighted by Psi^-1 (notes Trditev 4.1). B: everything free, from A's solution.
    """
    n = problem.n
    frozen = np.zeros(len(z0), dtype=bool)
    for d in frozen_components:
        frozen[d * n : (d + 1) * n] = True
    if frozen.any():
        z0, _ = _least_squares(problem, z0, ~frozen, sigma, z_lo, z_hi, minimizer_kwargs)
    return _least_squares(problem, z0, np.ones(len(z0), dtype=bool), sigma, z_lo, z_hi, minimizer_kwargs)


def _laplace_factor(problem, z, sigma):
    """Lower Cholesky factor R of the Gauss-Newton Hessian H = J^T J at z (H^-1 = R^-T R^-1 is the
    Laplace covariance, notes §5.5), with a tiny ridge so that a flat direction (a symmetry, notes
    Trditev 5.3) gives a huge variance instead of a failure."""
    _, J = problem.model_jacobian(z)
    J = np.vstack([problem.data_jacobian(z, sigma), J])
    H = J.T @ J
    return _cholesky_jitter(H, np.trace(H) / len(H))


# ---------------------------------------------------------------------------------- bayes ----


def _hmc(problem, z_map, sigma_map, R, sample_sigma, lo, hi, n_iter, n_leapfrog, step_size, burn_in,
         target_accept, rng):
    """
    Hamiltonian Monte Carlo on -log posterior (notes §3.8), in whitened coordinates
    z = z_map + R^-T w (R from the Laplace approximation at the MAP), so that the posterior is close
    to a standard normal and one scalar step size suits every coordinate. The step size is adapted
    during burn-in towards `target_accept` (MAGI aims at 60-90 %). θ outside its bounds is rejected.

    With sample_sigma, the noise std of every data channel is sampled too, as η = log σ with a flat
    prior on σ.
    """
    n_z = len(z_map)
    channels = problem.counts > 0
    counts = problem.counts[channels]
    rows = [s for s, keep in zip(problem.slices, channels) if keep]
    eta_map = np.log(sigma_map[channels]) if sample_sigma else np.empty(0)
    eta_scale = 1.0 / np.sqrt(2.0 * counts) if sample_sigma else np.empty(0)

    def to_params(w):
        z = z_map + solve_triangular(R.T, w[:n_z], lower=False)
        sigma = sigma_map.copy()
        if sample_sigma:
            sigma[channels] = np.exp(eta_map + eta_scale * w[n_z:])
        return z, sigma

    def energy_grad(w):
        z, sigma = to_params(w)
        theta = z[problem.n * problem.D :]
        if np.any(theta < lo) or np.any(theta > hi):
            return np.inf, None
        r_model, J_model = problem.model_jacobian(z)
        r_data = problem.data_residuals(z, sigma)
        if not (np.all(np.isfinite(r_model)) and np.all(np.isfinite(r_data))):
            return np.inf, None
        U = 0.5 * (r_model @ r_model + r_data @ r_data)
        grad_z = J_model.T @ r_model + problem.data_jacobian(z, sigma).T @ r_data
        grad = solve_triangular(R, grad_z, lower=True)  # chain rule through z = z_map + R^-T w
        if sample_sigma:
            # data term in η = log σ: sum((g - y)^2) e^{-2η} / 2 + (n_c - 1) η (flat prior on σ)
            sigma_c = sigma[channels]
            sq = np.array([np.sum((r_data[s] * sc) ** 2) for s, sc in zip(rows, sigma_c)])
            eta = np.log(sigma_c)
            U += np.sum((counts - 1) * eta)
            grad = np.concatenate([grad, eta_scale * (-sq * np.exp(-2 * eta) + counts - 1)])
        return U, grad

    dim = n_z + (int(channels.sum()) if sample_sigma else 0)
    w = np.zeros(dim)
    U, g = energy_grad(w)
    n_burn = int(burn_in * n_iter)
    log_step = np.log(step_size)
    samples_z, samples_sigma, accepted = [], [], 0
    for it in range(n_iter):
        eps = np.exp(log_step) * rng.uniform(0.8, 1.2)  # jittered, so trajectories don't resonate
        q = rng.standard_normal(dim)
        w_new, q_new = w.copy(), q - 0.5 * eps * g
        U_new, g_new = U, g
        for step in range(n_leapfrog):
            w_new = w_new + eps * q_new
            U_new, g_new = energy_grad(w_new)
            if not np.isfinite(U_new):
                break
            q_new = q_new - (eps if step < n_leapfrog - 1 else 0.5 * eps) * g_new
        log_ratio = (U + 0.5 * q @ q) - (U_new + 0.5 * q_new @ q_new) if np.isfinite(U_new) else -np.inf
        accept_prob = float(np.exp(min(0.0, log_ratio)))
        if rng.uniform() < accept_prob:
            w, U, g = w_new, U_new, g_new
            accepted += it >= n_burn
        if it < n_burn:
            log_step += (accept_prob - target_accept) / np.sqrt(it + 1)
        else:
            z, sigma = to_params(w)
            samples_z.append(z)
            samples_sigma.append(sigma)
    n_kept = max(n_iter - n_burn, 1)
    return np.array(samples_z), np.array(samples_sigma), accepted / n_kept, float(np.exp(log_step))


# ------------------------------------------------------------------------------------ public ----


def magi(
    f: Callable,
    ts,
    xs,
    theta_0,
    unobserved: "Iterable[int] | None" = None,
    unobserved_inits=None,
    method: Literal["max_likelihood", "bayes"] = "max_likelihood",
    n_seeds: "int | None" = None,
    grid=None,
    ode_points=None,
    midpoints: bool = False,
    kernel: KernelKind = "matern52",
    phi=None,
    sigma=None,
    mu=None,
    beta: "float | None" = None,
    bounds=None,
    jac: "Callable | None" = None,
    observe: "Callable | None" = None,
    ys=None,
    gp_restarts: int = 5,
    minimizer_kwargs: "dict | None" = None,
    hmc_kwargs: "dict | None" = None,
    seed: "int | None" = None,
) -> OptimizeResult:
    """
    θ-parameter estimation for x' = f(x, t, θ) on data (ts, xs) with MAGI: a GP prior on every
    component, the ODE imposed on the GP's derivative, no numerical integration. Components
    without data are allowed. See PyBM notes/magi.md.

    Parameters
    ----------
    f : callable
        f(x, t, θ) -> dx/dt, numpy, batched: x (m, D), t (m,) -> (m, D). Must act pointwise (row i
        of the output depends only on row i of x).
    ts : array-like, shape (n,)
        Data times, increasing.
    xs : array-like, shape (n, D)
        Data on the states, NaN where missing. A column without any data is an unobserved
        component. With `observe`, xs only sets the prior and the starting values of the
        components it has data for (may be all NaN); the data is then `ys`.
    theta_0 : array-like, shape (p,)
        Initial guess for θ.
    unobserved : iterable of int, optional
        Columns of xs to treat as unobserved (their data is ignored).
    unobserved_inits : array-like, optional
        Starting values of the unobserved components: shape (n_unobserved,) (a constant) or
        (len(ts), n_unobserved), a trajectory on the data times (e.g. a simulation with θ_0; then
        their GP prior is fitted to it). Default: their prior mean (`mu`), else 0.
    method : "max_likelihood" or "bayes"
        "max_likelihood": the posterior's mode (MAP), by Gauss-Newton - fast. "bayes": samples from
        the posterior by HMC, started at the MAP (MAGI as in the paper; slower).
    n_seeds : int, optional
        Number of seeds - the points where the GP's values are unknowns - evenly spaced over ts.
        The unknowns are n_seeds * D + p, whatever the amount of data. Default: the seeds are
        `grid`.
    grid : array-like, optional
        Explicit seeds (ignored with n_seeds). Default: ts, i.e. MAGI as in the paper; a denser grid
        containing ts (`refine_grid`) enforces the ODE at more points.
    ode_points : array-like, optional
        Where the ODE is imposed. Default: the seeds if they contain every data time (MAGI as in
        the paper), else the data times.
    midpoints : bool
        Seeds and ODE points halfway between consecutive data times; the data term stays at the
        data times. For inputs that are piecewise constant between data times, see the module
        docstring. With n_seeds, the seeds are n_seeds evenly spaced midpoints and the ODE is imposed
        at every midpoint; without, every midpoint is a seed. Ignored for what `grid` or
        `ode_points` set explicitly.
    kernel : "matern52", "matern32" or "rbf"
        GP kernel. MAGI uses Matérn ν = 2.01; Matérn 5/2 is the closest closed form. RBF is smooth
        everywhere and biases sharp dynamics (notes §3.7).
    phi : list, optional
        Per component, None (fit from data / default) or (signal variance, length scale).
    sigma : array-like, optional
        Noise std per data channel (a component of xs, or a column of ys with `observe`). Default:
        from a GP fit of the data.
    mu : array-like, shape (D,), optional
        Constant prior mean per component. Default: the data mean; for unobserved components the
        mean of unobserved_inits, else 0.
    beta : float, optional
        Tempering of the GP terms (notes §3.5). Default D (n_seeds + n_ode_points) / (2 N), N the
        number of data points; D |I| / N for MAGI as in the paper.
    bounds : (lo, hi), optional
        Bounds on θ (flat prior inside).
    jac : callable, optional
        jac(x, t, θ) -> (df/dx (m, D, D), df/dθ (m, D, p)), analytic derivatives of f. Default:
        forward differences (1 + D + p calls of f per Jacobian).
    observe : callable, optional
        g(x, t, θ) -> (m, q): what the data measures, batched and pointwise like f, e.g. an output
        computed from the states. The data is then ys.
    ys : array-like, shape (n, q), optional
        The data for `observe`, NaN where missing.
    gp_restarts : int
        Optimizer restarts of the GP fits (each on at most 300 points).
    minimizer_kwargs : dict, optional
        Forwarded to `scipy.optimize.least_squares` (MAP).
    hmc_kwargs : dict, optional
        method="bayes": n_iter (2000), n_leapfrog (20), step_size (0.2, initial), burn_in (0.5),
        target_accept (0.75), sample_sigma (True unless `sigma` is given).
    seed : int, optional
        Seed of the HMC sampler.

    Returns
    -------
    scipy.optimize.OptimizeResult
        .theta, .x (n_seeds, D) at the seeds .grid, .x_ts (len(ts), D) the states at the data times,
        .ode_points, .sigma (per data channel), .phi (D, 2: signal variance, length scale), .mu,
        .beta, .theta_cov (Laplace covariance of θ at the MAP; valid when no bound is active and θ is
        identifiable), and the least-squares fields (success, status, message, nfev, cost).
        method="bayes" adds .theta_samples (n_kept, p), .x_samples (n_kept, n_seeds, D),
        .sigma_samples, .acceptance, .step_size and .map (the MAP result); .theta, .x and .sigma are
        then posterior means.

    Notes
    -----
    Before fitting, fix symmetries of the model: if scaling an unobserved component can be absorbed
    by a parameter (Lotka-Volterra with x2 hidden: x2 -> λ x2, b -> b / λ), the MAP slides to the
    bound of that parameter (notes Trditev 5.3).
    """
    sigma_given = sigma is not None
    problem, z0, z_lo, z_hi, frozen_components, phi, sigma, mu, beta = _setup(
        f, ts, xs, theta_0, unobserved, unobserved_inits, grid, n_seeds, ode_points, midpoints, kernel, phi, sigma,
        mu, beta, bounds, jac, gp_restarts, observe, ys,
    )
    z_map, lsq = _map(problem, z0, frozen_components, sigma, z_lo, z_hi, minimizer_kwargs or {})

    R = _laplace_factor(problem, z_map, sigma)
    n_x = problem.n * problem.D
    cov = cho_solve((R, True), np.eye(len(z_map)))
    U_map, theta_map = problem.unpack(z_map)
    common = dict(grid=problem.seeds, ode_points=problem.ode_points, phi=phi, mu=mu, beta=beta)
    result = OptimizeResult(
        theta=theta_map, x=U_map, x_ts=problem.states(U_map, "data"), sigma=sigma, theta_cov=cov[n_x:, n_x:],
        success=lsq.success, status=lsq.status, message=lsq.message, nfev=lsq.nfev, cost=lsq.cost, **common,
    )
    if method == "max_likelihood":
        return result
    if method != "bayes":
        raise ValueError(f"method must be 'max_likelihood' or 'bayes', got {method!r}.")

    options = {"n_iter": 2000, "n_leapfrog": 20, "step_size": 0.2, "burn_in": 0.5, "target_accept": 0.75,
               "sample_sigma": not sigma_given, **(hmc_kwargs or {})}
    sample_sigma = options.pop("sample_sigma")
    samples_z, samples_sigma, acceptance, step = _hmc(
        problem, z_map, sigma, R, sample_sigma, z_lo[n_x:], z_hi[n_x:], rng=np.random.default_rng(seed), **options
    )
    x_samples = samples_z[:, :n_x].reshape(len(samples_z), problem.D, problem.n).transpose(0, 2, 1)
    x_mean = x_samples.mean(0)
    return OptimizeResult(
        theta=samples_z[:, n_x:].mean(0), x=x_mean, x_ts=problem.states(x_mean, "data"), sigma=samples_sigma.mean(0),
        theta_samples=samples_z[:, n_x:], x_samples=x_samples, sigma_samples=samples_sigma,
        acceptance=acceptance, step_size=step, map=result, **common,
    )
