import torch
import gpytorch


def negative_log_predictive_density(model, likelihood, x, y):
    """
    Negative log predictive density of model (normalized by number of observations).
    """
    device = x.device
    y = y.to(device)
    model = model.to(device)
    likelihood = likelihood.to(device)

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(x))
        return -pred_dist.log_prob(y).item() / y.shape[0]
