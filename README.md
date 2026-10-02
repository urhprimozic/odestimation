**Experimental repo!**
It will be finished, ko se mi bo dal.  

## Instalation 
Run
```
pip install -e /path/to/this/folder/. 
```
## Gaussian processes
`odestimate.gp.regressor.GP` is a wrapper around sklearn.gaussian_process, which assign seperate hyperparameters to every variable.
```
from odestimate.gp.regressor import GP 
import numpy as np 

# create some data 
>>> t_obs = np.linspace(0, 1, 10)
>>> y_obs = np.sin(t_obs) * t_obs
>>> y_obs = y_obs.reshape((1,10)) # of shape (n_vars, n_time_points)
>>> gp = GP(t_obs, y_obs, kernel="rbf", n_restarts_optimizer=10) # use more than just one restart!
>>> # compute values at time=0.5
>>> t=0.5
>>> mean = gp(t)
>>> std = gp.std(t)
>>> derivative_mean = gp.derivative(t)
>>> derivative_std = gp.std_derivative(t)
>>> print(f"mean at time={t}: {mean} with std {std}")
>>> print(f"mean of derivative at time={t}: {derivative_mean} with std {derivative_std}")
mean at time=0.5: [[0.23971074]] with std [[8.70227814e-06]]
mean of derivative at time=0.5: [[0.918235]] with std [[6.69871869e-05]]
```

## Gradient matching
`odestimate.gradient_matching` estimates the parameters θ of an ODE x' = f(x, t, θ) without integrating it: a GP interpolates the data, and θ is fitted so that f matches the derivative of the GP.
```python
import numpy as np
import torch
from scipy.integrate import solve_ivp
from odestimate.gp.regressor import GP
from odestimate.gradient_matching import precompute, gradient_matching, confidence_interval

# data: noisy Lotka-Volterra at 20 time points, y_obs of shape (n_vars, n_time_points)
def lotka_volterra(t, x, a, b, c, d):
    return [a * x[0] - b * x[0] * x[1], -c * x[1] + d * x[0] * x[1]]

t_obs = np.linspace(0, 2, 20)
y_obs = solve_ivp(lotka_volterra, (0, 2), [5.0, 3.0], t_eval=t_obs, args=(2.0, 1.0, 4.0, 1.0)).y
y_obs = y_obs + np.random.default_rng(0).normal(scale=0.1, size=y_obs.shape)

# the model f(x, t, theta) -> dx/dt, batched over time points: x (m, n_vars), t (m,) -> (m, n_vars).
# Written in torch, which gives exact Jacobians (default engine="torch"); for a numpy f pass
# engine="scipy" to precompute, gradient_matching and confidence_interval.
def f(x, t, theta):
    a, b, c, d = theta
    x1, x2 = x[:, 0], x[:, 1]
    return torch.stack([a * x1 - b * x1 * x2, -c * x2 + d * x1 * x2], dim=1)

# 1. interpolate the data: gradient matching needs derivatives, so the interpolant is a GP
gp = GP(t_obs, y_obs, n_restarts_optimizer=10)

# 2. the GP at the points where the derivatives are matched - independent of f, so compute it once
#    and reuse it for every candidate model
ts = np.linspace(t_obs[0], t_obs[-1], 50)
xs, dxs, x_vars, dx_vars, covs = precompute(ts, gp)

# 3. fit theta (keyword arguments go to scipy.optimize.least_squares)
result = gradient_matching(ts, xs, dxs, f, theta_0=[1.5, 1.5, 3.0, 1.5], bounds=(0, 10))
print(result.x)  # ≈ [2.0, 1.0, 4.3, 1.1], true [2, 1, 4, 1]

# 4. optional: confidence interval of the gradient matching loss at the fitted theta - an interval
#    for the loss, used to compare candidate models, not for theta
lower, upper = confidence_interval(ts, xs, dxs, f, result.x, x_vars, dx_vars, covs, p_value=0.1)
```

## Multishooting
`odestimate.latent.multishooting.ms` integrates the ODE on `n_subintervals` subintervals, from states at the start of each subinterval that are fitted together with θ, with the trajectory required to be continuous across subintervals. Variables without data are allowed.
```python
from odestimate.latent.multishooting import ms

# t_obs, y_obs and f as above

# 1. the data at the times ts where the trajectory is compared with it. ms needs no derivatives, so
#    the interpolant can be anything: linear interpolation as here, a GP, a spline, a step function -
#    or none at all: ts = t_obs, xs = y_obs.T
ts = np.linspace(t_obs[0], t_obs[-1], 41)
xs = np.stack([np.interp(ts, t_obs, y) for y in y_obs], axis=1)  # (len(ts), n_vars)

# 2. fit theta. engine="torch" integrates with explicit Euler and a fixed step; engine="scipy" takes
#    a numpy f and any scipy.integrate.solve_ivp setting as a keyword (method, rtol, atol, ...)
result = ms(f, ts, xs, n_subintervals=4, theta_0=[1.5, 1.5, 3.0, 1.5], engine="torch", step=0.01)
print(result.theta)  # ≈ [2.0, 1.0, 4.2, 1.1]
print(result.inits)  # fitted states at the start of each subinterval, (n_subintervals, n_vars)

# a variable without data: NaN in its column, marked unobserved, with a guess of its value at ts[0]
xs_hidden = xs.copy()
xs_hidden[:, 1] = np.nan
result = ms(f, ts, xs_hidden, 4, [1.5, 1.5, 3.0, 1.5], unobserved=[1], unobserved_inits=[2.0],
            engine="torch", step=0.01)
# with x2 hidden only b * x2 is identifiable: scaling x2 by λ and b by 1/λ leaves x1 unchanged
print(result.theta[1] * result.inits[0, 1])  # ≈ 3.3, true b * x2(0) = 3
```
Other options of `ms`: `minimizer_kwargs` (forwarded to `scipy.optimize.minimize`, e.g. `{"options": {"maxiter": 500}}`), `weights` and `huber_delta` (scaling and robustness of the residuals), and `observe=g, ys=...` when the data measure a function g(x, t, θ) of the state instead of the state itself.