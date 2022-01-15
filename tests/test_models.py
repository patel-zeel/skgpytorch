import pytest
from bayesian_benchmarks.data import Boston

import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from skgpytorch.models import ExactGPRegressor
from .gpytorch_models import exact_gp_regressor_from_gpytorch


def test_exact_gp_regressor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_iters = 5

    data = Boston()
    data.X_train = torch.tensor(data.X_train, dtype=torch.float)
    data.Y_train = torch.tensor(data.Y_train, dtype=torch.float).ravel()
    data.X_test = torch.tensor(data.X_test, dtype=torch.float)
    data.Y_test = torch.tensor(data.Y_test, dtype=torch.float).ravel()
    kernels = [
        ScaleKernel(RBFKernel(ard_num_dims=data.X_train.shape[1])),
        ScaleKernel(MaternKernel(ard_num_dims=data.X_train.shape[1])),
    ]

    for seed in range(5):
        for kernel in kernels:
            gpytorch_model = exact_gp_regressor_from_gpytorch(
                data, kernel, seed, n_iters, device
            )
            gp = ExactGPRegressor(
                data.X_train,
                data.Y_train,
                kernel,
                device=device,
            )
            gp.fit(n_iters=n_iters, random_state=seed)
            pred_dist = gp.predict(data.X_train, data.Y_train, data.X_test)

            assert torch.allclose(pred_dist.variance, gpytorch_model.pred_dist.variance)
            assert torch.allclose(pred_dist.mean, gpytorch_model.pred_dist.mean)
