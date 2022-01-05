import torch
import gpytorch
from ._models import ExactGPModel
import warnings


class BaseRegressor:
    def __init__(self, train_x, train_y, kernel, random_state, device):
        self.train_x = train_x
        self.train_y = train_y
        self.kernel = kernel
        self.history = {"train_loss": []}
        self.device = device
        if random_state is not None:
            torch.manual_seed(random_state)

        self.train_x = self.train_x.to(self.device)
        self.train_y = self.train_y.to(self.device)
        self.model = self.model.to(device)
        self.likelihood = self.likelihood.to(device)

    def fit(self, n_iters=1, n_restarts=1, verbose=0, verbose_gap=1):
        self.model.train()
        self.likelihood.train()
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        if len(self.history["train_loss"]) > 0:
            warnings.warn(
                "Training loss is not empty. This may cause unexpected behavior."
            )

        best_loss = float("inf")
        for restart in range(n_restarts):
            history = {"train_loss": []}
            for param in self.model.parameters():
                torch.nn.init.normal_(param, mean=0, std=0.1)

            for i in range(n_iters):
                self.optimizer.zero_grad()
                output = self.model(self.train_x)
                loss = -self.mll(output, self.train_y)
                loss.backward()
                if verbose and i % verbose_gap == 0:
                    print(
                        "Restart: {}, Iter: {}, Loss: {:.4f}, Best Loss: {:.4f}".format(
                            restart, i, loss.item(), best_loss
                        )
                    )
                history["train_loss"].append(loss.item())
                self.optimizer.step()

            # Last loss
            output = self.model(self.train_x)
            loss = -self.mll(output, self.train_y)

            # Check if best loss
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = self.model.state_dict()
                best_likelihood_state = self.likelihood.state_dict()
                # self.likelihood.save_state_dict()
                self.history = history

            # Load the best model
            self.model.load_state_dict(best_model_state)
            self.likelihood.load_state_dict(best_likelihood_state)

    def predict(self, x, dist_only=False):
        x = x.to(self.device)
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

    def __init__(self, train_x, train_y, kernel, random_state=None, device="cpu"):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(train_x, train_y, self.likelihood, kernel)
        super().__init__(train_x, train_y, kernel, random_state, device)
