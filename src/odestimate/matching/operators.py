"""
The linear operators of `odestimate.matching`, as sparse matrices, and pointwise derivatives.

Every residual of the matching methods is linear in values of f (or of the states) at some points:
- `linear_interpolation`: free states, linear between seeds, at any points, and their slopes.
- `integral_windows`: integrals of f over windows by the trapezoid rule (integral matching).
- `ode_rule`: the constraint of the free states, one per gap between seeds: the slope over the gap
  minus the mean of f at the gap's ODE points.
- `ode_windows`: the same in integral form over longer windows: x(b) - x(a) = ∫_a^b f, cumulative
  from the first seed or of a given length.
"""

import numpy as np
from scipy import sparse


def gap(points, seeds):
    """Index k of the gap [seeds[k], seeds[k+1]) every point lies in. A point on a seed belongs to
    the gap that starts there, points outside the seeds to the first or last gap."""
    return np.clip(np.searchsorted(seeds, points, side="right") - 1, 0, len(seeds) - 2)


def linear_interpolation(points, seeds):
    """
    Sparse matrices V, S of shape (len(points), len(seeds)): for values u at the seeds, V @ u are
    the values at `points` of the piecewise linear function through (seeds, u), S @ u its slopes.
    Outside the seeds the first and last pieces continue.
    """
    k = gap(points, seeds)
    h = seeds[k + 1] - seeds[k]
    w = (points - seeds[k]) / h
    rows = np.repeat(np.arange(len(points)), 2)
    cols = np.stack([k, k + 1], axis=1).ravel()
    shape = (len(points), len(seeds))
    V = sparse.csr_matrix((np.stack([1 - w, w], axis=1).ravel(), (rows, cols)), shape=shape)
    S = sparse.csr_matrix((np.stack([-1 / h, 1 / h], axis=1).ravel(), (rows, cols)), shape=shape)
    return V, S


def ode_rule(ode_points, seeds):
    """
    The constraint of the free states, g = S @ U - A @ F, F = f at the ODE points: per gap with an
    ODE point, the slope of the states over the gap minus the mean of f at its ODE points - the
    mean of f over the gap, so x_{k+1} - x_k = ∫ f over the gap (with the gap's midpoint, the implicit
    midpoint rule). Returns S (n_constraints, n_seeds) and A (n_constraints, n_ode_points), A None when
    no gap has more than one ODE point (then the constraints are the ODE points', in their order).
    """
    k = gap(ode_points, seeds)
    counts = np.bincount(k, minlength=len(seeds) - 1)
    if counts.max() <= 1:
        return linear_interpolation(ode_points, seeds)[1], None
    used = np.flatnonzero(counts)
    _, S = linear_interpolation((seeds[used] + seeds[used + 1]) / 2, seeds)
    row = np.full(len(seeds) - 1, -1)
    row[used] = np.arange(len(used))
    A = sparse.csr_matrix((1 / counts[k], (row[k], np.arange(len(ode_points)))), shape=(len(used), len(ode_points)))
    return S, A


def ode_windows(ode_points, seeds, window):
    """
    The constraint of the free states in integral form over windows of whole gaps, one ending at every
    seed after the first: x(b) - x(a) = ∫_a^b f, divided by b - a, as g = S @ U - A @ F. Every gap's
    integral is its length times the mean of f at its ODE points (with the centres of equal parts, the
    composite midpoint rule, which keeps f off the seeds, where inputs may jump).

    Parameters
    ----------
    ode_points, seeds : np.ndarray
        Every gap between seeds needs an ODE point.
    window : "cumulative" or float
        "cumulative": a = seeds[0]. A length w: a = the first seed at or after b - w (at least one gap).

    Returns
    -------
    S : (n_seeds - 1, n_seeds), A : (n_seeds - 1, n_ode_points), sparse.
    """
    k = gap(ode_points, seeds)
    counts = np.bincount(k, minlength=len(seeds) - 1)
    if np.any(counts == 0):
        raise ValueError("every gap between seeds needs an ODE point for the constraint over windows")
    h = np.diff(seeds)
    G = sparse.csr_matrix((h[k] / counts[k], (k, np.arange(len(ode_points)))), shape=(len(h), len(ode_points)))
    end = np.arange(1, len(seeds))
    if window == "cumulative":
        start = np.zeros_like(end)
    else:
        start = np.minimum(np.searchsorted(seeds, seeds[end] - float(window), side="left"), end - 1)
    rows = np.repeat(np.arange(len(end)), end - start)
    cols = np.concatenate([np.arange(a, b) for a, b in zip(start, end)])
    C = sparse.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(end), len(h)))  # gaps in every window
    L = seeds[end] - seeds[start]
    S = sparse.csr_matrix((np.r_[-1 / L, 1 / L], (np.r_[np.arange(len(end)), np.arange(len(end))], np.r_[start, end])),
                          shape=(len(end), len(seeds)))
    return S, sparse.diags(1 / L) @ C @ G

def integral_windows(points, window=None, n_quad: int = 10):
    """
    Windows [a_j, b_j] ending at the matching points, and the trapezoid rule over them.

    Parameters
    ----------
    points : array-like
        Matching points (sorted and made unique here).
    window : None, "cumulative" or float
        None: between consecutive points. "cumulative": from the first point to every other one.
        A float w: [b - w, b] for every point b after the first (not before the first point).
    n_quad : int
        Trapezoid steps between consecutive points.

    Returns
    -------
    nodes : np.ndarray
        Quadrature nodes, sorted: every point, every window start and n_quad - 1 nodes between
        consecutive points.
    W : scipy.sparse.csr_matrix, shape (n_windows, n_nodes)
        W @ g ≈ the integral of g over every window, from g at the nodes.
    start, end : np.ndarray
        Index of every window's start and end in `nodes`.
    """
    points = np.unique(np.asarray(points, float))
    if window is None:
        a, b = points[:-1], points[1:]
    elif window == "cumulative":
        a, b = np.full(len(points) - 1, points[0]), points[1:]
    else:
        b = points[1:]
        a = np.maximum(b - float(window), points[0])
    steps = (points[:-1, None] + np.diff(points)[:, None] * (np.arange(n_quad) / n_quad)).ravel()
    nodes = np.unique(np.concatenate([steps, points, a]))
    start, end = np.searchsorted(nodes, a), np.searchsorted(nodes, b)
    h = np.diff(nodes)
    rows, cols, values = [], [], []
    for j, (i, k) in enumerate(zip(start, end)):
        w = np.zeros(k - i + 1)
        w[:-1] += h[i:k] / 2
        w[1:] += h[i:k] / 2
        rows.append(np.full(len(w), j))
        cols.append(np.arange(i, k + 1))
        values.append(w)
    W = sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(len(a), len(nodes)))
    return nodes, W, start, end


def block_diagonal(blocks):
    """Sparse block-diagonal matrix with one (a, b) block per point, from blocks of shape (m, a, b)."""
    m, a, b = blocks.shape
    rows = np.broadcast_to((np.arange(m)[:, None] * a + np.arange(a))[:, :, None], blocks.shape)
    cols = np.broadcast_to((np.arange(m)[:, None] * b + np.arange(b))[:, None, :], blocks.shape)
    return sparse.csr_matrix((blocks.ravel(), (rows.ravel(), cols.ravel())), shape=(m * a, m * b))


def pointwise_derivatives(g, X, t, theta, G):
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
