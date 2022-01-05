# skgpytorch
[![Coverage Status](https://coveralls.io/repos/github/patel-zeel/skgpytorch/badge.svg?branch=main)](https://coveralls.io/github/patel-zeel/skgpytorch?branch=main)

GPyTorch Models in Scikit-learn wrapper.

# Example

```py
import torch
from skgpytorch.models import ExactGPRegressor
from skgpytorch.metrics import mean_squared_error, negative_log_predictive_density
from gpytorch.kernels import RBFKernel, ScaleKernel

# Define a model
train_x = torch.rand(10, 1)
train_y = torch.rand(10)
test_x = torch.rand(10, 1)
test_y = torch.rand(10)

kernel = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))
gp = ExactGPRegressor(train_x, train_y, kernel, random_state=0, device="cpu")

# Fit the model
gp.fit(n_iters=10, verbose=True, n_restarts=2, verbose_gap=2)

# Get the predictions
# f_mean, f_var = gp.predict(test_x)
# OR
pred_dist = gp.predict(test_x, dist_only=True)

# Calculate metrics
print("MSE:", mean_squared_error(pred_dist, test_x, test_y))
print("NLPD:", negative_log_predictive_density(pred_dist, test_x, test_y))
```

```bash
Restart: 0, Iter: 0, Loss: 1.0135, Best Loss: inf
Restart: 0, Iter: 2, Loss: 0.9371, Best Loss: inf
Restart: 0, Iter: 4, Loss: 0.8644, Best Loss: inf
Restart: 0, Iter: 6, Loss: 0.7978, Best Loss: inf
Restart: 0, Iter: 8, Loss: 0.7382, Best Loss: inf
Restart: 1, Iter: 0, Loss: 0.9626, Best Loss: 0.6819
Restart: 1, Iter: 2, Loss: 0.8948, Best Loss: 0.6819
Restart: 1, Iter: 4, Loss: 0.8239, Best Loss: 0.6819
Restart: 1, Iter: 6, Loss: 0.7537, Best Loss: 0.6819
Restart: 1, Iter: 8, Loss: 0.6880, Best Loss: 0.6819
MSE: 0.08736331760883331
NLPD: 0.49492106437683103
```