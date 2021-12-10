import pytest

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

    data, kernel, n_iters = load_data_and_params()

    for seed in range(5):
        gpytorch_model = exact_gp_regressor_from_gpytorch(data, kernel, seed, n_iters)
        gp = ExactGPRegressor(data.X_train, data.Y_train, kernel, random_state=seed)
        gp.fit(data.X_train, data.Y_train, n_iters=n_iters)

        a = negative_log_predictive_density(
            gpytorch_model.model, gpytorch_model.likelihood, data.X_test, data.Y_test
        )
        b = negative_log_predictive_density(
            gp.model, gp.likelihood, data.X_test, data.Y_test
        )
        assert torch.allclose(a, b)


def test_nlpd_value():
    data, kernel, n_iters = load_data_and_params()

    n_iters = 100

    gp = ExactGPRegressor(data.X_train, data.Y_train, kernel, random_state=0)
    gp.fit(data.X_train, data.Y_train, n_iters=n_iters)

    res = negative_log_predictive_density(
        gp.model, gp.likelihood, data.X_test, data.Y_test
    )

    assert res < 0
