# Shared Lotka-Volterra / Protein Transduction fixtures for the odestimate test suite.
#
# Same two systems (and same true theta/initial conditions/sampling) Wenk et al. (2019) use to
# benchmark gradient matching, and the ones PyBM's own benchmark suite replicates - reused here so
# results are comparable, not because the tests import anything from PyBM (they don't - odestimate
# is a standalone reimplementation).
import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

DTYPE = torch.float64

# ---------------------------------------------------------------- Lotka-Volterra ----

LV_THETA = np.array([2.0, 1.0, 4.0, 1.0])  # theta1, theta2, theta3, theta4
LV_IC = np.array([5.0, 3.0])
LV_T = np.linspace(0.0, 2.0, 20)
LV_NOISE_STD = 0.1


def lv_true_trajectory() -> np.ndarray:
    """(2, n) - noiseless x1(t), x2(t) at LV_T."""
    t1, t2, t3, t4 = LV_THETA

    def rhs(t, y):
        x1, x2 = y
        return [t1 * x1 - t2 * x1 * x2, -t3 * x2 + t4 * x1 * x2]

    sol = solve_ivp(rhs, (LV_T[0], LV_T[-1]), LV_IC, t_eval=LV_T, rtol=1e-10, atol=1e-12)
    return sol.y


def lv_f(x: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """x'(t) = f(x,t,theta) for LV, torch, x: (m,2) -> (m,2)."""
    x1, x2 = x[:, 0], x[:, 1]
    t1, t2, t3, t4 = theta[0], theta[1], theta[2], theta[3]
    dx1 = t1 * x1 - t2 * x1 * x2
    dx2 = -t3 * x2 + t4 * x1 * x2
    return torch.stack([dx1, dx2], dim=1)


@pytest.fixture
def lv_data():
    """`(t, x_true, x_noisy)` - `x_true`/`x_noisy` shape `(2, 20)` (the `(n_vars, n)` convention
    `odestimate.gp.regressor.GP` uses)."""
    x_true = lv_true_trajectory()
    rng = np.random.default_rng(0)
    x_noisy = x_true + rng.normal(scale=LV_NOISE_STD, size=x_true.shape)
    return LV_T, x_true, x_noisy


# ---------------------------------------------------------- Protein Transduction ----

PT_THETA = np.array([0.07, 0.6, 0.05, 0.3, 0.017, 0.3])  # theta1..theta6
PT_IC = np.array([1.0, 0.0, 1.0, 0.0, 0.0])  # S, Sd, R, RS, Rpp
PT_T = np.array([0, 1, 2, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100], dtype=float)
PT_NOISE_STD = 0.001


def pt_true_trajectory() -> np.ndarray:
    """(5, n) - noiseless S, Sd, R, RS, Rpp at PT_T."""
    t1, t2, t3, t4, t5, t6 = PT_THETA

    def rhs(t, y):
        S, Sd, R, RS, Rpp = y
        return [
            -t1 * S - t2 * S * R + t3 * RS,
            t1 * S,
            -t2 * S * R + t3 * RS + t5 * Rpp / (t6 + Rpp),
            t2 * S * R - t3 * RS - t4 * RS,
            t4 * RS - t5 * Rpp / (t6 + Rpp),
        ]

    sol = solve_ivp(rhs, (PT_T[0], PT_T[-1]), PT_IC, t_eval=PT_T, method="Radau", rtol=1e-10, atol=1e-12)
    return sol.y


def pt_f(x: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """x'(t) = f(x,t,theta) for PT, torch, x: (m,5) -> (m,5)."""
    S, Sd, R, RS, Rpp = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    t1, t2, t3, t4, t5, t6 = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]
    dS = -t1 * S - t2 * S * R + t3 * RS
    dSd = t1 * S
    dR = -t2 * S * R + t3 * RS + t5 * Rpp / (t6 + Rpp)
    dRS = t2 * S * R - t3 * RS - t4 * RS
    dRpp = t4 * RS - t5 * Rpp / (t6 + Rpp)
    return torch.stack([dS, dSd, dR, dRS, dRpp], dim=1)


@pytest.fixture
def pt_data():
    """`(t, x_true, x_noisy)` - `x_true`/`x_noisy` shape `(5, 15)`."""
    x_true = pt_true_trajectory()
    rng = np.random.default_rng(0)
    x_noisy = x_true + rng.normal(scale=PT_NOISE_STD, size=x_true.shape)
    return PT_T, x_true, x_noisy
