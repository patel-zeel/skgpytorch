import torch
import gpytorch


def negative_log_predictive_density(dist, y):
    """
    Negative log predictive density of model (normalized by number of observations).
    """
    return -dist.log_prob(y.ravel().to(dist.mean.device)).item() / y.shape[0]


def mean_squared_error(dist, y, squared=True):
    """
    Mean Squared Error
    """
    mse = torch.square(y.ravel().to(dist.mean.device) - dist.mean).mean().item()
    if not squared:
        return mse ** 0.5  # Root mean square error
    return mse
