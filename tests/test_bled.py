from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from bled_model import PROBMOT_THETA, load_bled, make_f, make_scipy_rhs
from odestimate.gradient_matching import gradient_matching

BLED_PATH = Path(__file__).parent / "data" / "96.data"


def test_gradient_matching_bled():
    """
    One real (not synthetic) Bled season, single file (96.data) - deliberately just one, so there
    are no cross-season gaps to worry about (see the chat this followed: pooling multiple season
    files leaves multi-month gaps with no data, which is its own separate problem).
    """
    t, phyto, temp, light, daph = load_bled(BLED_PATH)
    f = make_f(torch.as_tensor(temp), torch.as_tensor(light), torch.as_tensor(daph))

    result = gradient_matching(t, phyto[None, :], f, theta_0=PROBMOT_THETA)

    assert result.success
    assert np.all(np.isfinite(result.x))
    # NOT asserting the constants stay positive/near ProBMoT's own values: checked directly, this
    # unconstrained fit lands `respRate` slightly negative (~-5e-4, i.e. "no respiration") and
    # `halfSaturation` at ~1e5 (i.e. "no grazing limitation") - a real practical-identifiability
    # finding on this real dataset (same flavor as PT's own "sloppiness" - see test_gradient_
    # matching_pt), not a bug: gradient matching here has no bounds/regularization to rule out a
    # nearby, comparably-good-fitting but physically-degenerate direction.

    # Forward-simulate with the fitted theta and check it stays in the right ballpark of the real
    # trajectory - gradient matching never integrates the ODE itself, so this is the first time
    # the fit is checked against an actual simulated trajectory, not just the GP-implied gradient.
    temp_i = interp1d(t, temp, fill_value="extrapolate")
    light_i = interp1d(t, light, fill_value="extrapolate")
    daph_i = interp1d(t, daph, fill_value="extrapolate")
    rhs = make_scipy_rhs(temp_i, light_i, daph_i, result.x)
    sol = solve_ivp(rhs, (t[0], t[-1]), [phyto[0]], t_eval=t)

    assert sol.success
    rmse = float(np.sqrt(np.mean((sol.y[0] - phyto) ** 2)))
    # phyto.conc spans ~0.5-4.3 on this dataset; ProBMoT's own (much more sophisticated, bounded,
    # multi-dataset) fit reaches RMSE~0.25 - checked directly, this plain, unconstrained, single-
    # season gradient-matching fit reaches ~1.3, meaningfully worse but not a blow-up. `< 2.0` is a
    # loose bound that only catches genuine divergence, not "doesn't match ProBMoT's own quality".
    assert rmse < 2.0
