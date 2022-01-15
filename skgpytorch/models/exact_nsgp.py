import torch
from .base import BaseRegressor
from nsgptorch.models import GP
import warnings


class ExactNSGPRegressor(BaseRegressor):
    """[summary]
    Call the constructor of base class after defining the model.
    """

    def __init__(self, kernel_list, input_dim, inducing_points, device="cpu"):
        super().__init__()
        self.device = device
        self.model = GP(kernel_list, input_dim, inducing_points)

    def forward(self, X, y):
        return self.model(X, y)

    def predict(self, X_orig, y_orig, X_test):
        return self.model.predict(
            X_orig.to(self.device), y_orig.to(self.device), X_test.to(self.device)
        )
