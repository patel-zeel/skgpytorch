# skgpytorch
[![Coverage Status](https://coveralls.io/repos/github/patel-zeel/skgpytorch/badge.svg?branch=main)](https://coveralls.io/github/patel-zeel/skgpytorch?branch=main)

[GPyTorch](https://gpytorch.ai/) Models in [Scikit-learn](https://scikit-learn.org/stable/) wrapper.

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
gp = ExactGPRegressor(train_x, train_y, kernel)

# Fit the model (This supports batch training of GP models as well)
gp.fit(n_epochs=2, verbose=True, n_restarts=1, verbose_gap=2, batch_size=10, lr=0.1, random_state=0)

# Get the predictions
pred_dist = gp.predict(test_x)

# Access properties of predictive distribution
pred_mean = pred_dist.mean # Mean
pred_var = pred_dist.variance # Variance
pred_stddev = pred_dist.stddev # Standard deviation
lower, upper = pred_dist.confidence_region() # 95% confidence region

# Calculate metrics (Soon this will be implemented in gpytorch itself)
print("MSE:", mean_squared_error(pred_dist, test_y))
print("NLPD:", negative_log_predictive_density(pred_dist, test_y))
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
