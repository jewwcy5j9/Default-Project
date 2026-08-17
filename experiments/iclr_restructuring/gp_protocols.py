"""Locked Gaussian-process protocols used by the paper analyses."""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


PRIMARY_GP_PROTOCOL = {
    "id": "GP-primary-0.05-v1",
    "kernel": "1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.05)",
    "alpha": 1e-4,
    "normalize_y": True,
    "n_restarts_optimizer": 1,
    "random_state": 0,
    "clr_clip": 1e-6,
    "feature_scaling": "training-fold StandardScaler",
}

T5_DIAGNOSTIC_GP_PROTOCOL = {
    "id": "GP-T5-diagnostic-0.01-v1",
    "display_name": "GP-T5-diagnostic",
    "kernel": "RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)",
    "alpha": 1e-4,
    "normalize_y": True,
    "n_restarts_optimizer": 0,
    "random_state": 0,
}


def make_primary_gp() -> GaussianProcessRegressor:
    return GaussianProcessRegressor(
        kernel=1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.05),
        alpha=1e-4,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=0,
    )


def make_t5_diagnostic_gp() -> GaussianProcessRegressor:
    return GaussianProcessRegressor(
        kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01),
        alpha=1e-4,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=0,
    )
