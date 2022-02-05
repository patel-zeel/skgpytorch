import pytest
from bayesian_benchmarks.data import Boston

import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from skgpytorch.models import ExactGPRegressor
from .gpytorch_models import exact_gp_regressor_from_gpytorch
from skgpytorch.metrics import negative_log_predictive_density, mean_squared_error


@pytest.mark.comp_model
def test_exact_gp_regressor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    n_iters = 5

    data = Boston()
    data.X_train = torch.tensor(data.X_train, dtype=torch.float).to(device)
    data.Y_train = torch.tensor(data.Y_train, dtype=torch.float).ravel().to(device)
    data.X_test = torch.tensor(data.X_test, dtype=torch.float).to(device)
    data.Y_test = torch.tensor(data.Y_test, dtype=torch.float).ravel().to(device)
    kernels = [
        ScaleKernel(RBFKernel(ard_num_dims=data.X_train.shape[1])),
    ]

    seed = 0
    for kernel in kernels:
        gpytorch_model = exact_gp_regressor_from_gpytorch(
            data, kernel, seed, n_iters, device
        )
        gp = ExactGPRegressor(
            data.X_train,
            data.Y_train,
            kernel,
        )
        gp.to(device)
        gp.fit(n_epochs=n_iters, random_state=seed)
        pred_dist = gp.predict(data.X_test)

        # avg_var = (pred_dist.variance - gpytorch_model.pred_dist.variance).abs_().mean()
        # avg_mean = (pred_dist.mean - gpytorch_model.pred_dist.mean).abs_().mean()
        # print(avg_mean, avg_var)
        # assert avg_var < 0.5
        # assert avg_mean < 0.5

        rmse_a = mean_squared_error(pred_dist, data.Y_test, squared=False)
        rmse_b = mean_squared_error(
            gpytorch_model.pred_dist, data.Y_test, squared=False
        )
        print(rmse_a, rmse_b)
        assert abs(rmse_a - rmse_b) < 0.01

        nlpd_a = negative_log_predictive_density(pred_dist, data.Y_test)
        nlpd_b = negative_log_predictive_density(gpytorch_model.pred_dist, data.Y_test)
        print(nlpd_a, nlpd_b)
        assert abs(nlpd_a - nlpd_b) < 0.01
