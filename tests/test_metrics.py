import pytest

from bayesian_benchmarks.data import Boston

import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from skgpytorch.models import ExactGPRegressor
from skgpytorch.metrics import negative_log_predictive_density
from .gpytorch_models import exact_gp_regressor_from_gpytorch


def load_data_and_params():
    data = Boston()
    data.X_train = torch.tensor(data.X_train, dtype=torch.float)
    data.Y_train = torch.tensor(data.Y_train, dtype=torch.float).ravel()
    data.X_test = torch.tensor(data.X_test, dtype=torch.float)
    data.Y_test = torch.tensor(data.Y_test, dtype=torch.float).ravel()
    kernel = ScaleKernel(RBFKernel(ard_num_dims=data.X_train.shape[1]))
    n_iters = 10
    return data, kernel, n_iters


def test_nlpd_equal():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, kernel, n_iters = load_data_and_params()

    for seed in range(5):
        gpytorch_model = exact_gp_regressor_from_gpytorch(
            data, kernel, seed, n_iters, device
        )
        gp = ExactGPRegressor(
            data.X_train, data.Y_train, kernel, random_state=seed, device=device
        )
        gp.fit(n_iters=n_iters)

        a = negative_log_predictive_density(
            gp.predict(data.X_test.to(device), dist_only=True),
            data.X_test.to(device),
            data.Y_test,
        )
        b = negative_log_predictive_density(
            gpytorch_model.pred_dist, data.X_test.to(device), data.Y_test
        )
        assert abs(a - b) < 1e-5
