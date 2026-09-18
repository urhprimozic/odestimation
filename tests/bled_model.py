"""
Model #939 from ProBMoT's own exhaustive search on the Bled lake dataset
(examples/bled/probmot/BledInducedBest939.pbm in the PyBM repo) - the phyto.conc growth /
respiration / grazing equation, hand-derived directly from ProBMoT's process templates
(examples/bled/probmot/AquaticEcosystem.pbl) - NOT by importing/reusing any PyBM code (odestimate
is a standalone reimplementation).

Structure ProBMoT's search picked (only phyto.conc - the one state variable fit here; all three
nutrient limitation terms it chose are "NoNutrientLim" = 1, i.e. inert, so nutrients drop out
entirely):

    growth:      td(phyto) += maxGrowthRate * (temp/refTempGrowth) * lightLim * phyto
                     lightLim = light * exp(-light/optLight + 1) / optLight      (OptimalLightLim)
    respiration: td(phyto) += -respRate * phyto^2                               (Temp2RespirationPP,
                                                                                   no temp limit)
    grazing:     td(phyto) += -maxFiltrationRate * (temp/refTempDaph) * daph * phyto * phytoLim
                     phytoLim = phyto^2 / (phyto^2 + halfSaturation)             (Monod2PhytoLim)

7 free constants (theta, in this order): maxGrowthRate, refTempGrowth, optLight, respRate,
maxFiltrationRate, refTempDaph, halfSaturation. `temp`, `light`, `daph` are exogenous (observed
drivers, never fit).

Two callers need this equation in two DIFFERENT calling conventions, which is why the module
splits into a `phyto_rhs` core plus two thin builders:
- `odestimate.gradient_matching` needs `f(x,t,theta)`, batched over the FIXED observation grid -
  since the drivers are already observed at exactly those times, no interpolation is needed, and
  (important) none is allowed: `jacfwd` forbids calling `.numpy()` on ANY tensor while its
  transform is active, even one that doesn't depend on `theta` - so `make_f` closes over
  already-torch, pre-aligned driver tensors instead of interpolating on the fly.
- `scipy.integrate.solve_ivp` needs `(t,y) -> dy/dt` at ARBITRARY continuous `t` (its adaptive
  stepper doesn't stay on the data grid) and works entirely outside any torch transform, so
  `make_scipy_rhs` interpolates (`scipy.interpolate.interp1d`) freely.
"""
from pathlib import Path

import numpy as np
import torch

THETA_NAMES = [
    "maxGrowthRate", "refTempGrowth", "optLight", "respRate",
    "maxFiltrationRate", "refTempDaph", "halfSaturation",
]

# ProBMoT's own fitted values for this exact structure (BledInducedBest939.pbm) - a reference
# point (and a sane starting guess), not a "ground truth" - there is no ground truth for real data.
PROBMOT_THETA = np.array([
    0.6516029900666023, 22.0, 140.42919645642743, 0.006043472507553232,
    15.0, 12.091141569787668, 20.0,
])


def load_bled(path: "str | Path") -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Reads one Bled `.data` file. Returns `(t, phyto, temp, light, daph)`. `t` is rebased to
    start at 0 - the raw column is an absolute day count (~729028) - harmless to shift, avoids
    carrying large numbers through everything downstream for no reason."""
    data = np.genfromtxt(path, names=True)
    t = data["t"].astype(float)
    t = t - t[0]
    return t, data["phyto"], data["temp"], data["light_m"], data["daph_lit"]


def phyto_rhs(phyto: torch.Tensor, temp: torch.Tensor, light: torch.Tensor, daph: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """The core equation - every argument broadcastable torch tensors of the same shape. Returns
    `dphyto/dt`, same shape as `phyto`."""
    max_growth, ref_temp_growth, opt_light, resp_rate, max_filtration, ref_temp_daph, half_sat = theta

    light_lim = light * torch.exp(-light / opt_light + 1) / opt_light
    growth = max_growth * (temp / ref_temp_growth) * light_lim * phyto
    respiration = resp_rate * phyto**2
    phyto_lim = phyto**2 / (phyto**2 + half_sat)
    grazing = max_filtration * (temp / ref_temp_daph) * daph * phyto * phyto_lim

    return growth - respiration - grazing


def make_f(temp: torch.Tensor, light: torch.Tensor, daph: torch.Tensor):
    """
    Builds `f(x,t,theta) -> dx/dt` for `odestimate.gradient_matching` - `x`: `(m,1)`, `t`: `(m,)`
    (accepted only to match the shared `f(x,t,theta)` signature, unused - see module docstring).

    Parameters
    ----------
    temp, light, daph : torch.Tensor, shape (m,)
        The exogenous drivers, ALREADY evaluated at the same `m` times `x`/`t` will be called
        with (typically `t_obs` itself) - fixed, precomputed once, never re-interpolated here.
    """

    def f(x: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return phyto_rhs(x[:, 0], temp, light, daph, theta).unsqueeze(1)

    return f


def make_scipy_rhs(temp_interp, light_interp, daph_interp, theta):
    """
    Builds `(t,y) -> dy/dt` for `scipy.integrate.solve_ivp`, at a FIXED `theta`.

    Parameters
    ----------
    temp_interp, light_interp, daph_interp : callable, t (float) -> value (float)
        E.g. `scipy.interpolate.interp1d(t_full, values, fill_value="extrapolate")` - evaluated at
        `solve_ivp`'s own adaptive-step query times, which don't stay on the data grid.
    theta : array-like, shape (7,)
    """
    theta_t = torch.as_tensor(theta, dtype=torch.float64)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        phyto = torch.as_tensor(y, dtype=torch.float64)
        temp = torch.as_tensor(temp_interp(t), dtype=torch.float64)
        light = torch.as_tensor(light_interp(t), dtype=torch.float64)
        daph = torch.as_tensor(daph_interp(t), dtype=torch.float64)
        dphyto = phyto_rhs(phyto, temp, light, daph, theta_t)
        return dphyto.detach().cpu().numpy().reshape(-1)

    return rhs
