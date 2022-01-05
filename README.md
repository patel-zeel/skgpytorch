# skgpytorch
[![Coverage Status](https://coveralls.io/repos/github/patel-zeel/skgpytorch/badge.svg?branch=main)](https://coveralls.io/github/patel-zeel/skgpytorch?branch=main)

GPyTorch Models in Scikit-learn wrapper.

# Examples

```py
from skgpytorch.models import ExactGPRegressor
from skgpytorch.metrics import mean_squared_error, negative_log_predictive_density

# Define a model
gp = ExactGPRegressor(train_x, train_y, kernel, random_state=0, device="cpu")

# Fit the model
gp.fit(n_iters=10, verbose=True, n_restarts=5, verbose_gap=2)

# Get the predictions
# f_mean, f_var = gp.predict(test_x)
# OR
pred_dist = gp.predict(test_y, dist_only=True)

# Calculate metrics
print("MSE:", mean_squared_error(pred_dist, test_x, test_y))
print("NLPD:", negative_log_predictive_density(pred_dist, test_x, test_y))
```