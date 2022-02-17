# skgpytorch
[![Coverage Status](https://coveralls.io/repos/github/patel-zeel/skgpytorch/badge.svg?branch=main)](https://coveralls.io/github/patel-zeel/skgpytorch?branch=main)

[GPyTorch](https://gpytorch.ai/) Models in [Scikit-learn](https://scikit-learn.org/stable/) wrapper.

# Examples

## Simple GP Regression

```py
from skgpytorch.models import ExactGPRegressor
from gpytorch.kernels import RBFKernel, ScaleKernel

kernel = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))
model = ExactGPRegressor(train_x, train_y, kernel)
model.fit(n_epochs=10, lr=0.1)
pred_dist = gp.predict(test_x)
```

## Simple Sparse GP Regression

```py
from skgpytorch.models import SGPRegressor
from gpytorch.kernels import RBFKernel, ScaleKernel

kernel = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))
inducing_points = train_x[:10]
model = SGPRegressor(train_x, train_y, kernel, inducing_points)
model.fit(n_epochs=10, lr=0.1)
pred_dist = gp.predict(test_x)
```

## Simple Stochastic Variational GP Regression

```py
from skgpytorch.models import SVGPRegressor
from gpytorch.kernels import RBFKernel, ScaleKernel

kernel = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))
inducing_points = train_x[:10]
model = SVGPRegressor(train_x, train_y, kernel, inducing_points)
model.fit(n_epochs=10, lr=0.1)
pred_dist = gp.predict(test_x)
```

## Simple GP Regression (extended version)

```py
import torch
from skgpytorch.models import ExactGPRegressor
from skgpytorch.metrics import mean_squared_error, negative_log_predictive_density
from gpytorch.kernels import RBFKernel, ScaleKernel

# Define a model
train_x = torch.rand(12, 1)
train_y = torch.rand(12)
test_x = torch.rand(12, 1)
test_y = torch.rand(12)

kernel = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))
model = ExactGPRegressor(train_x, train_y, kernel)

# Fit the model (This supports batch training of GP models as well)
model.fit(n_epochs=2, verbose=True, n_restarts=1, verbose_gap=2, batch_size=6, lr=0.1, random_state=0)

# Access loss history
restart = 0
epoch_loss = model.history['epoch_loss'][restart]
iter_loss = model.history['iter_loss'][restart]
print('Epoch loss:', *map(lambda x: round(x, 2), epoch_loss))
print('Iter loss:', *map(lambda x: round(x, 2), iter_loss))

# Get the predictions
pred_dist = model.predict(test_x)

# Access properties of predictive distribution
pred_mean = pred_dist.mean # Mean
pred_var = pred_dist.variance # Variance
pred_stddev = pred_dist.stddev # Standard deviation
lower, upper = pred_dist.confidence_region() # 95% confidence region

# Calculate metrics (Soon this will be implemented in gpytorch itself)
print("MSE:", mean_squared_error(pred_dist, test_y))
print("NLPD:", negative_log_predictive_density(pred_dist, test_y))
```
Output
```py
restart: 0, epoch: 1, iter: 2, loss: 1.9412
restart: 0, epoch: 2, iter: 2, loss: 1.7454
Epoch loss: 0.97 0.87
Iter loss: 1.0 0.94 0.89 0.86
MSE: 0.0687108263373375
NLPD: 0.6836201349894205
```
