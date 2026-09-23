# gaussian processes + gradient matching
# interpolate the data using a GP -> interpolant x(t)
# compare x'(t) with f(x(t), t, theta) to estimate theta
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.stats import t as student_t
from torch.func import jacfwd, jacrev, vmap

from odestimate.gp.regressor import GP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64


def gradient_matching(t_obs, x_obs, f, theta_0, gp=None, **kwargs):
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
        A GP already fit elsewhere - reuse it instead of fitting a new one on `x_obs`. Lets a
        caller fit the GP ONCE on the full dataset (best possible interpolant) and then run
        gradient matching on a DIFFERENT (e.g. smaller, randomly sampled) set of `t_obs` query
        points - exactly what `uniform_trust_region` below needs.
    **kwargs : dict
        Additional keyword arguments to pass to `scipy.optimize.least_squares`.

    Returns
    -------
    scipy.optimize.OptimizeResult
        `.x` holds the estimated theta.
    """
    if gp is None:
        gp = GP(t_obs, x_obs)

    # The GP interpolant and its derivative don't depend on theta - compute them ONCE, as fixed
    # torch tensors, instead of re-querying the GP on every residual/gradient evaluation below.
    x_hat = torch.as_tensor(gp(t_obs), dtype=DTYPE, device=DEVICE)
    x_hat_deriv = torch.as_tensor(gp.derivative(t_obs), dtype=DTYPE, device=DEVICE)
    t = torch.as_tensor(t_obs, dtype=DTYPE, device=DEVICE)

    def residual(theta: torch.Tensor) -> torch.Tensor:
        return (x_hat_deriv - f(x_hat, t, theta)).reshape(-1)

    def fun(theta_np: np.ndarray) -> np.ndarray:
        theta = torch.as_tensor(theta_np, dtype=DTYPE, device=DEVICE)
        return residual(theta).detach().cpu().numpy()

    def jac(theta_np: np.ndarray) -> np.ndarray:
        theta = torch.as_tensor(theta_np, dtype=DTYPE, device=DEVICE)
        return jacfwd(residual)(theta).detach().cpu().numpy()

    return least_squares(fun, x0=theta_0, jac=jac, **kwargs)


def confidence_interval(gp, f, t_samples, theta, p_value):
    """
    Computes confidence intervals for the gradient matching loss 
            L_n(\hat \theta) +- t_{n-1}(1 - p_value / 2) * SE
    where SE = sqrt(Var[A] + Var[B]) is the standard error of the loss 
    where 
     - A: the variance of the L2 loss due to subsampling 
     - B:the variance of the L2 loss due to the GP's own uncertainty in x_hat and x_hat_deriv
    """
    n = len(t_samples)
    x_hat = torch.as_tensor(gp(t_samples), dtype=DTYPE, device=DEVICE)  # (n, d)
    x_hat_deriv = torch.as_tensor(gp.derivative(t_samples), dtype=DTYPE, device=DEVICE)  # (n, d)
    t = torch.as_tensor(t_samples, dtype=DTYPE, device=DEVICE)

    r = x_hat_deriv - f(x_hat, t, theta)  # (n, d), raw residuals r_i
    rho = (r**2).sum(dim=1)  # (n,), L2 penalty rho_i = ||r_i||^2
    L_n = float(rho.mean())

    rho_np = rho.detach().cpu().numpy()
    s2 = float(np.var(rho_np, ddof=1)) if n > 1 else 0.0  # Vir A: unbiased sample variance
    var_A = s2 / n

    # F_x(t_i, theta): (n, d, d), the RHS's own Jacobian w.r.t. its state argument, at each point
    # independently (a per-point, not cross-point, Jacobian - same vmap(jacrev(...)) pattern used
    # throughout PyBM for exactly this quantity).
    def f_single(x_i, t_i):
        return f(x_i.unsqueeze(0), t_i.unsqueeze(0), theta).squeeze(0)

    F_x = vmap(jacrev(f_single, argnums=0))(x_hat, t)  # (n, d, d)

    sigma2 = torch.as_tensor(gp.std(t_samples) ** 2, dtype=DTYPE, device=DEVICE)  # (n, d)
    sigma_d2 = torch.as_tensor(gp.std_derivative(t_samples) ** 2, dtype=DTYPE, device=DEVICE)  # (n, d)
    cross = torch.as_tensor(gp.cov_state_derivative(t_samples), dtype=DTYPE, device=DEVICE)  # (n, d)

    # Cov[delta r_i] = Sigma_d - F_x.Sigma_c - Sigma_c.F_x^T + F_x.Sigma.F_x^T (Sigma's diagonal -
    # state variables are modeled by independent GP1dims, see GP's own docstring).
    Sigma = torch.diag_embed(sigma2)  # (n, d, d)
    Sigma_d = torch.diag_embed(sigma_d2)  # (n, d, d)
    Sigma_c = torch.diag_embed(cross)  # (n, d, d), symmetric (diagonal)

    Fx_Sc = torch.bmm(F_x, Sigma_c)
    cov_dr = Sigma_d - Fx_Sc - Fx_Sc.transpose(1, 2) + torch.bmm(torch.bmm(F_x, Sigma), F_x.transpose(1, 2))

    # Var[delta rho_i] = 4 r_i^T Cov[delta r_i] r_i (delta rho = 2r . delta r for rho=||r||^2),
    # clamped at 0 - cov_dr is PSD in exact arithmetic (it's A.M.A^T for the genuine, PSD joint GP
    # covariance M), floating point can dip a hair negative.
    var_drho = torch.clamp(4.0 * torch.einsum("ni,nij,nj->n", r, cov_dr, r), min=0.0)  # (n,)
    var_B = float(var_drho.sum()) / n**2

    se = float(np.sqrt(var_A + var_B))
    t_star = float(student_t.ppf(1 - p_value / 2, df=max(n - 1, 1)))
    return L_n - t_star * se, L_n + t_star * se


def uniform_trust_region(t_obs, x_obs, f, theta_0, n_samples, p_value=0.1,gp=None, **kwargs):
    """
    Fast gradient matching for screening/pruning candidate `f`'s: fits the GP on the FULL data
    (best possible interpolant), then runs gradient matching on only `n_samples` time points
    sampled UNIFORMLY AT RANDOM from the observed range - plus a confidence interval for how far
    the resulting loss can plausibly be from the "full" (continuum, true-trajectory) loss. See
    `notes/gradient-matching-pruning.md` (PyBM repo) for the full derivation; not meant as a final
    fit, only to rank/eliminate candidates quickly before spending real effort on the survivors.

    Parameters
    ----------
    t_obs, x_obs : as in `gradient_matching` - the FULL observed data. The GP is fit on all of it.
    f : as in `gradient_matching`.
    theta_0 : as in `gradient_matching`.
    n_samples : int
        Number of test points, drawn i.i.d. uniformly from `[min(t_obs), max(t_obs)]` - genuinely
        continuous query points into the already-fitted GP, NOT a subset of `t_obs`'s own indices.
    p_value : float, optional
        Significance level alpha for the resulting `(1 - alpha)` confidence interval. Default
        `0.1` (90% CI).
    gp : odestimate.gp.regressor.GP, optional
        A GP already fit elsewhere - reuse it instead of fitting a new one on `x_obs.
    **kwargs : dict
        Forwarded to `scipy.optimize.least_squares`.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Same as `gradient_matching`'s own return value, fit on the `n_samples` sampled points.
    interval : tuple[float, float]
        `(lower, upper)` confidence bound on the L2 loss achievable with the full data and the
        true (not interpolated) trajectory - see `confidence_interval`.
    """
    # sample points
    t_obs = np.asarray(t_obs, dtype=float)
    t_samples = np.random.default_rng().uniform(t_obs.min(), t_obs.max(), size=n_samples)
    # fit gp
    if gp is None:
        gp = GP(t_obs, x_obs)
    
    # run gradient matching on sampled points 
    result = gradient_matching(t_samples, None, f, theta_0, gp=gp, **kwargs)

    theta = torch.as_tensor(result.x, dtype=DTYPE, device=DEVICE)
    # return confidence interval 
    interval = confidence_interval(gp, f, t_samples, theta, p_value)
    return result, interval