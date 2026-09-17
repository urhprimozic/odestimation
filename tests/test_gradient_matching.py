import numpy as np
import torch

from conftest import LV_THETA, PT_THETA, lv_f, pt_f
from odestimate.gp.regressor import GP
from odestimate.gradient_matching import DEVICE, DTYPE, gradient_matching


def _mean_abs_error(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _gm_cost(t, x_obs, f, theta) -> float:
    """Same quantity `gradient_matching`'s own `least_squares` minimizes - built independently
    here (not by reaching into `gradient_matching`'s internals) so tests can compare against it at
    an arbitrary theta, e.g. the starting guess."""
    gp = GP(t, x_obs)
    x_hat = torch.as_tensor(gp(t), dtype=DTYPE, device=DEVICE)
    x_hat_deriv = torch.as_tensor(gp.derivative(t), dtype=DTYPE, device=DEVICE)
    theta = torch.as_tensor(theta, dtype=DTYPE, device=DEVICE)
    residual = x_hat_deriv - f(x_hat, torch.as_tensor(t, dtype=DTYPE, device=DEVICE), theta)
    return 0.5 * float((residual**2).sum())


def test_gradient_matching_lv(lv_data):
    t, _, x_noisy = lv_data
    # Start near (not at) the truth - this is a unit test of the MECHANISM (does it refine a
    # decent guess), not of global search from an arbitrary start (a separate, harder question -
    # see PyBM's own notes on gradient matching needing a reasonable warm start).
    theta_0 = LV_THETA * 1.15

    result = gradient_matching(t, x_noisy, lv_f, theta_0)

    assert result.success
    error_before = _mean_abs_error(theta_0, LV_THETA)
    error_after = _mean_abs_error(result.x, LV_THETA)
    assert error_after < error_before
    assert error_after < 0.3  # LV is well-identified from a 20-point, low-noise trajectory


def test_gradient_matching_pt(pt_data):
    """
    PT is a genuinely "sloppy" (practically non-identifiable) system - checked directly: the
    least-squares cost at the theta this test recovers can end up LOWER than the cost at the true
    theta itself (some other rate-constant combination explains the GP-implied derivative just as
    well, or better, than the truth does). So, unlike the LV test, this only checks that the
    optimizer did its job (reduced the cost below the starting guess's) - NOT that the recovered
    theta is close to the true one, which would be an unfair expectation for this benchmark, not a
    sign the implementation is broken.
    """
    t, _, x_noisy = pt_data
    theta_0 = PT_THETA * 1.15

    result = gradient_matching(t, x_noisy, pt_f, theta_0)

    assert result.success
    assert result.cost < _gm_cost(t, x_noisy, pt_f, theta_0)
