import torch
import gpytorch
from .base import BaseRegressor
from .gpytorch_models import ExactGPModel
import warnings


class ExactGPRegressor(BaseRegressor):
    """[summary]
    Call the constructor of base class after defining the model.
    """

    def __init__(self, train_x, train_y, kernel):
        super().__init__()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.train_x = train_x
        self.train_y = train_y
        self.model = ExactGPModel(self.train_x, self.train_y, likelihood, kernel)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )

    def forward(self, X, y):
        self.model.set_train_data(X, y, strict=False)
        output = self.model(X)
        return -self.mll(output, y)

    def predict(self, X_test):
        self.model.set_train_data(
            inputs=self.train_x, targets=self.train_y, strict=False
        )
        if len(self.history["train_loss"]) < 1:
            warnings.warn(
                "Model is not fitted yet. This may cause unexpected behavior."
            )

        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.pred_dist = self.model.likelihood(self.model(X_test))
            return self.pred_dist
