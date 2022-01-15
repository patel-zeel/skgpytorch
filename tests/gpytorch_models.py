import torch
import gpytorch

# Test ExactGPRegressor
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class exact_gp_regressor_from_gpytorch:
    def __init__(self, data, kernel, seed, n_iters, device):
        # initialize likelihood and model
        torch.manual_seed(seed)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(data.X_train, data.Y_train, self.likelihood, kernel)

        data.X_train = data.X_train.to(device)
        data.Y_train = data.Y_train.to(device)
        self.model = self.model.to(device)
        self.likelihood = self.likelihood.to(device)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.1
        )  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for param in self.model.parameters():
            torch.nn.init.normal_(param, mean=0.0, std=1.0)
            # print("old", param)

        for _ in range(n_iters):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(data.X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, data.Y_train)
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.pred_dist = self.likelihood(self.model(data.X_test.to(device)))
