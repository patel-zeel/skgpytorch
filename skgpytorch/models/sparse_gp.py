import gpytorch
from .base import BaseRegressor


class SGPRModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, inducing_points):
        super(SGPRModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = kernel
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            kernel, inducing_points=inducing_points, likelihood=likelihood
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SGPRegressor(BaseRegressor):
    """[summary]
    Call the constructor of base class after defining the model.
    """

    def __init__(self, train_x, train_y, kernel, inducing_points):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SGPRModel(train_x, train_y, likelihood, kernel, inducing_points)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        super().__init__(train_x, train_y, mll)
