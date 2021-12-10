import torch
import gpytorch


def negative_log_predictive_density(model, likelihood, x, y):
    """
    Negative log predictive density of model (normalized by number of observations).
    """
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(x))
        return -pred_dist.log_prob(y) / y.shape[0]
