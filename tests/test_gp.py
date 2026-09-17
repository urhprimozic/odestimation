import numpy as np

from odestimate.gp.regressor import GP, GP1dim


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


# ---------------------------------------------------------------- GP1dim, on LV x1 ----


def test_gp1dim_mean_denoises_better_than_raw_noise(lv_data):
    t, x_true, x_noisy = lv_data
    gp = GP1dim(t, x_noisy[0], n_restarts_optimizer=3)

    mean_mse = _mse(gp.mean(t), x_true[0])
    raw_mse = _mse(x_noisy[0], x_true[0])
    # smoothing should get meaningfully closer to the truth than the raw noisy data already is
    assert mean_mse < raw_mse


def test_gp1dim_derivative_matches_finite_difference(lv_data):
    t, x_true, x_noisy = lv_data
    gp = GP1dim(t, x_noisy[0], n_restarts_optimizer=3)

    # central finite difference of the TRUE (noiseless) trajectory, interior points only
    fd = (x_true[0][2:] - x_true[0][:-2]) / (t[2:] - t[:-2])
    d_hat = gp.derivative(t)[1:-1]
    assert _mse(d_hat, fd) < 1.0  # loose - GP derivative is smoothed, FD is not


def test_gp1dim_std_and_cov_are_well_formed(lv_data):
    t, _, x_noisy = lv_data
    gp = GP1dim(t, x_noisy[0], n_restarts_optimizer=3)

    std = gp.std(t)
    assert std.shape == t.shape
    assert np.all(std >= 0) and not np.any(np.isnan(std))

    cov = gp.cov(t)
    assert cov.shape == (len(t), len(t))
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-8  # PSD up to numerical noise

    dstd = gp.std_derivative(t)
    assert np.all(dstd >= 0) and not np.any(np.isnan(dstd))

    dcov = gp.cov_derivative(t)
    assert np.allclose(dcov, dcov.T)
    assert np.linalg.eigvalsh(dcov).min() > -1e-8


def test_gp1dim_std_not_floored_at_noise_var(lv_data):
    """See the chat this followed: `GP1dim` deliberately does NOT floor std/cov at noise_var
    internally anymore - that's left to whichever caller needs it. Densely-observed LV (20 points
    over a short, smooth span) should let the raw posterior std dip below noise_var somewhere."""
    t, _, x_noisy = lv_data
    gp = GP1dim(t, x_noisy[0], n_restarts_optimizer=3)
    assert np.any(gp.std(t) < np.sqrt(gp.noise_var()))


# ------------------------------------------------------------- GP (multi-variable) ----


def test_gp_multivar_shapes_and_recovery_lv(lv_data):
    t, x_true, x_noisy = lv_data
    gp = GP(t, x_noisy, n_restarts_optimizer=3)

    assert len(gp) == 2
    assert gp.noise_var().shape == (2,)
    assert gp.length_scale().shape == (2,)

    mean = gp.mean(t)
    assert mean.shape == (len(t), 2)
    for i in range(2):
        assert _mse(mean[:, i], x_true[i]) < _mse(x_noisy[i], x_true[i])

    assert gp.std(t).shape == (len(t), 2)
    assert gp.cov(t).shape == (2, len(t), len(t))
    assert gp.derivative(t).shape == (len(t), 2)
    assert gp.std_derivative(t).shape == (len(t), 2)
    assert gp.cov_derivative(t).shape == (2, len(t), len(t))


def test_gp_multivar_shapes_pt(pt_data):
    t, x_true, x_noisy = pt_data
    gp = GP(t, x_noisy, n_restarts_optimizer=3)

    assert len(gp) == 5
    mean = gp.mean(t)
    assert mean.shape == (len(t), 5)
    # PT is sparsely sampled (15 points over [0,100]) and near-noiseless - just check the GP
    # tracks the true trajectory reasonably, not to a tight tolerance.
    for i in range(5):
        assert _mse(mean[:, i], x_true[i]) < 1.0
