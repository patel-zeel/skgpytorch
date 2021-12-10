import torch
import gpytorch
from ._models import ExactGPModel
import warnings


class BaseRegressor:
    def __init__(self, train_x, train_y, kernel, random_state):
        self.train_x = train_x
        self.train_y = train_y
        self.kernel = kernel
        self.history = {"train_loss": []}
        if random_state is not None:
            torch.manual_seed(random_state)

    def fit(self, x, y, n_iters):
        self.model.train()
        self.likelihood.train()
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        if len(self.history["train_loss"]) > 0:
            warnings.warn(
                "Training loss is not empty. This may cause unexpected behavior."
            )
        self.history["train_loss"] = []

        for param in self.model.parameters():
            torch.nn.init.normal_(param, mean=0, std=0.1)
            print("new", param)

        for _ in range(n_iters):
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = -self.mll(output, y)
            loss.backward()
            self.history["train_loss"].append(loss.item())
            self.optimizer.step()

    def predict(self, x, dist_only=False):
        if len(self.history["train_loss"]) < 1:
            warnings.warn(
                "Model is not fitted yet. This may cause unexpected behavior."
            )

        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.pred_dist = self.likelihood(self.model(x))

        if dist_only:
            return self.pred_dist

        return self.pred_dist.mean, self.pred_dist.variance


class ExactGPRegressor(BaseRegressor):
    """[summary]
    Call the constructor of base class after defining the model.
    """

    def __init__(self, train_x, train_y, kernel, random_state=None):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(train_x, train_y, self.likelihood, kernel)
        super().__init__(train_x, train_y, kernel, random_state)
