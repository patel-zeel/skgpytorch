import pytest

from bayesian_benchmarks.data import Boston

import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from nsgptorch.kernels import rbf
from skgpytorch.models import ExactGPRegressor, ExactNSGPRegressor
from skgpytorch.metrics import negative_log_predictive_density, mean_squared_error
from .gpytorch_models import exact_gp_regressor_from_gpytorch


def load_data_and_params():
    data = Boston()
    data.X_train = torch.tensor(data.X_train, dtype=torch.float)
    data.Y_train = torch.tensor(data.Y_train, dtype=torch.float).ravel()
    data.X_test = torch.tensor(data.X_test, dtype=torch.float)
    data.Y_test = torch.tensor(data.Y_test, dtype=torch.float).ravel()
    kernel = ScaleKernel(RBFKernel(ard_num_dims=data.X_train.shape[1]))
    n_iters = 100
    return data, kernel, n_iters


def test_metrics_nsgp():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # print(f"device:{device}")
    data, kernel, n_iters = load_data_and_params()

    for seed in range(5):
        gpytorch_model = exact_gp_regressor_from_gpytorch(
            data, kernel, seed, n_iters, device
        )
        gp = ExactNSGPRegressor(
            [rbf for i in range(data.X_train.shape[1])],
            data.X_train.shape[1],
            [None for i in range(data.X_train.shape[1])],
            device=device,
        )
        gp.fit(
            data.X_train,
            data.Y_train,
            n_iters=n_iters,
            random_state=seed,
        )

        pred_dist = gp.predict(data.X_train, data.Y_train, data.X_test)

        nlpd_a = negative_log_predictive_density(pred_dist, data.Y_test)
        nlpd_b = negative_log_predictive_density(gpytorch_model.pred_dist, data.Y_test)
        assert abs(nlpd_a - nlpd_b) < 0.5

        mse_a = mean_squared_error(pred_dist, data.Y_test, squared=False)
        mse_b = mean_squared_error(gpytorch_model.pred_dist, data.Y_test, squared=False)

        assert abs(mse_a - mse_b) < 0.5


def test_metrics_sgp():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # print(f"device:{device}")
    data, kernel, n_iters = load_data_and_params()

    for seed in range(5):
        gpytorch_model = exact_gp_regressor_from_gpytorch(
            data, kernel, seed, n_iters, device
        )
        gp = ExactGPRegressor(data.X_train, data.Y_train, kernel, device=device)
        gp.fit(n_iters=n_iters, random_state=seed)

        pred_dist = gp.predict(data.X_train, data.Y_train, data.X_test)

        nlpd_a = negative_log_predictive_density(pred_dist, data.Y_test)
        nlpd_b = negative_log_predictive_density(gpytorch_model.pred_dist, data.Y_test)
        assert abs(nlpd_a - nlpd_b) < 1e-5

        mse_a = mean_squared_error(pred_dist, data.Y_test)
        mse_b = mean_squared_error(gpytorch_model.pred_dist, data.Y_test)

        assert abs(mse_a - mse_b) < 1e-5
