# gaussian processes + gradient matching
# interpolate the data using a GP -> interpolant x(t)
# compare x'(t) with f(x(t), t, theta) to estimate theta
import numpy as np
import torch
from scipy.optimize import least_squares
from torch.func import jacfwd

from odestimate.gp.regressor import GP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64


def gradient_matching(t_obs, x_obs, f, theta_0, **kwargs):
    """
    Gradient matching for parameter estimation in ODEs using Gaussian Processes.
    Finds the theta that best explains the ODE system

        x'(t) = f(x(t), t, theta)

    by matching the GP's own derivative against f, instead of actually integrating the ODE.

    Parameters
    ----------
    t_obs : array-like, shape (n_samples,)
        The observed time points.
    x_obs : array-like, shape (n_features, n_samples)
        The observed state values, one row per state variable (same convention as
        `odestimate.gp.regressor.GP`).
    f : callable
        The ODE right-hand side, f(x, t, theta) -> dx/dt. Must be written in torch (it gets
        differentiated w.r.t. theta).
    theta_0 : array-like
        Initial guess for theta.
    **kwargs : dict
        Additional keyword arguments to pass to `scipy.optimize.least_squares`.

    Returns
    -------
    scipy.optimize.OptimizeResult
        `.x` holds the estimated theta.
    """
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
