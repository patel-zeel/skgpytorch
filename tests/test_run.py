import torch
from gpytorch.kernels import ScaleKernel, RBFKernel
from skgpytorch.models import ExactGPRegressor, SVGPRegressor, SGPRegressor

n = 128
d = 5
n_inducing = 16
batch_sz = 64
n_epochs = 5
x = torch.randn(n, d)
y = torch.randn(n)
x_test = torch.randn(n * 2, d)


def get_exact_gp():
    kernel = ScaleKernel(RBFKernel(ard_num_dims=x.shape[1]))
    model = ExactGPRegressor(x, y, kernel)
    return model


def get_sgpr():
    kernel = ScaleKernel(RBFKernel(ard_num_dims=x.shape[1]))
    model = SGPRegressor(x, y, kernel, inducing_points=x[:n_inducing])
    return model


def get_svgp():
    kernel = ScaleKernel(RBFKernel(ard_num_dims=x.shape[1]))
    model = SVGPRegressor(x, y, kernel, inducing_points=x[:n_inducing])
    return model


def test_all_models():
    for model in [get_svgp(), get_exact_gp(), get_sgpr()]:
        model.fit(n_epochs=n_epochs)
        model.fit(n_epochs=n_epochs, batch_size=64)
        model.fit(n_epochs=n_epochs, n_restarts=2)
        pred_dist = model.predict(x_test)
        assert pred_dist.mean.shape == (x_test.shape[0],)
