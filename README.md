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
>>> y_obs.reshape((1,10)) # of shape (n_vars, n_time_points)
>>> gp = GP(t_obs, y_obs, kernel="rbf")
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