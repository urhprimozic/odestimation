"""
θ-parameter estimation for x' = f(x, t, θ) without integrating the ODE: every component is either
matched to an interpolant of its data (gradient or integral matching), or free - its values at seeds
unknowns and its equation a constraint (collocation). One estimator, `core.matching`; the methods
below are it with the two choices set:

                        every component matched   observed matched, the rest free   every component free
    derivative form     gradient_matching         gm_collocation                    collocation
    integral form       integral_matching         im_collocation

(Collocation's constraint is in integral form already: x_{k+1} - x_k = ∫ f over every gap.)
`odestimate.gradient_matching` is the older gradient matching on precomputed states and derivatives,
with torch and confidence intervals, for screening.
"""

from odestimate.matching.core import matching


def gradient_matching(f, ts, xs, theta_0, **kwargs):
    """
    Gradient matching: every component is an interpolant x̂ of its data (a GP by default, or `gp=`),
    and θ minimizes ½ Σ_i |(x̂'(t_i) - f(x̂(t_i), t_i, θ)) / s|^power at the matching points t_i
    (`points` / `n_points`, default ts). Parameters: see `core.matching`.
    """
    return matching(f, ts, xs, theta_0, matched="all", form="derivative", **kwargs)


def integral_matching(f, ts, xs, theta_0, **kwargs):
    """
    Integral matching: every component is an interpolant x̂ of its data (a GP by default, or `gp=`),
    and θ minimizes ½ Σ_j |(x̂(b_j) - x̂(a_j) - ∫_{a_j}^{b_j} f(x̂(s), s, θ) ds) / s|^power over windows
    ending at the matching points (`points` / `n_points`, default ts; `window`, `n_quad`). No
    derivative of the interpolant. Parameters: see `core.matching`.
    """
    return matching(f, ts, xs, theta_0, matched="all", form="integral", **kwargs)


def collocation(f, ts, xs, theta_0, **kwargs):
    """
    Collocation: every component free - its values at the seeds unknowns, linear in between - with the
    constraint x_{k+1} - x_k = ∫ f over every gap between seeds (by default f at the gap's midpoint: the
    implicit midpoint rule), and the loss on the data. With n seeds the constraint leaves θ and the
    initial state to the data. Parameters: see `core.matching`.
    """
    return matching(f, ts, xs, theta_0, matched="none", **kwargs)


def gm_collocation(f, ts, xs, theta_0, **kwargs):
    """
    Gradient matching on the components with data, collocation for the others (driven by the
    interpolant of the observed ones), plus a data term for outputs computed from the states
    (`observe`). Without unobserved components gradient matching, with no observed ones collocation.
    Parameters: see `core.matching`.
    """
    return matching(f, ts, xs, theta_0, matched="observed", form="derivative", **kwargs)


def im_collocation(f, ts, xs, theta_0, **kwargs):
    """`gm_collocation` with integral matching on the components with data. Parameters: see `core.matching`."""
    return matching(f, ts, xs, theta_0, matched="observed", form="integral", **kwargs)
