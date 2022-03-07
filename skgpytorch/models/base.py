from gc import callbacks
from mimetypes import init
import torch
import faiss


class BaseRegressor(torch.nn.Module):
    def __init__(self, train_x, train_y, mll):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.mll = mll

        self.best_restart = None

    def _init_params(self, init_fn=None):
        if init_fn is None:
            init_fn = lambda param: torch.nn.init.normal_(param)

        for param in self.mll.parameters():
            init_fn(param)

    def _compute_nn_idx(self, xd, xq, k):
        """
        Not using GPU version because of the limits (max 2048 points)
        xd: training data
        xq: query data
        """
        self.cpu_index = faiss.IndexFlatL2(xd.size(-1))
        xd = (xd.data.float()).cpu().numpy()
        self.cpu_index.add(xd)
        return torch.from_numpy(self.cpu_index.search(xq, k)[1]).long()

    def loss_func(self, X, y):
        if self.mll.__class__.__name__ != "VariationalELBO":
            self.mll.model.set_train_data(X, y, strict=False)

        output = self.mll.model(X)
        return -self.mll(output, y)

    def predict(self, test_x, include_likelihood=True):
        return self._predict(self.train_x, self.train_y, test_x, include_likelihood)

    def fit(
        self,
        batch_size=None,
        lr=0.1,
        n_epochs=1,
        n_restarts=1,
        random_state=None,
        callback_fn=None,
        init_fn=None,
    ):
        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)

        # Batch settings
        if batch_size is None:
            batch_size = self.train_x.shape[0]
        assert batch_size > 0 and batch_size <= self.train_x.shape[0]
        batch_mode = False if batch_size == self.train_x.shape[0] else True
        if batch_mode:
            train_nn_idx = self._compute_nn_idx(
                xd=self.train_x, xq=self.train_x, k=batch_size
            )

        # Intialize batch data same as training data
        X_batch = self.train_x
        y_batch = self.train_y

        # Initialize optimizer
        best_loss = float("inf")
        best_mll_state = None
        self.optimizer = torch.optim.Adam(self.mll.parameters(), lr=lr)

        n_iters = max(1, self.train_x.shape[0] // batch_size)
        self.mll.train()
        for restart in range(n_restarts):
            if restart > 0:  # Don't reset the model if it's the first restart
                self.init_params(init_fn)
            for epoch in range(n_epochs):
                loss = 0
                for iteration in range(n_iters):
                    if batch_mode:
                        idx = torch.randint(
                            low=0, high=self.train_x.shape[0], size=(1,)
                        )[0]
                        indices = train_nn_idx[idx]
                        X_batch = self.train_x[indices]
                        y_batch = self.train_y[indices]

                    self.optimizer.zero_grad()

                    batch_loss = self.loss_func(X_batch, y_batch)
                    batch_loss.backward()
                    loss += batch_loss.item()
                    if callback_fn is not None:
                        callback_fn()

                    self.optimizer.step()

            # Check if best loss
            loss = loss / n_iters
            if loss < best_loss:
                self.best_restart = restart
                best_loss = loss
                best_mll_state = self.mll.state_dict()

        # Load the best model
        if best_mll_state is not None:
            self.mll.load_state_dict(best_mll_state)

        if callback_fn is not None:
            callback_fn.finalize()

    def predict(self, X, include_likelihood=True):
        return self._predict(self.train_x, self.train_y, X, include_likelihood)

    def _predict(self, train_x, train_y, test_x, include_likelihood=True):
        if self.mll.__class__.__name__ != "VariationalELBO":
            self.mll.model.set_train_data(train_x, train_y, strict=False)

        self.mll.eval()
        if include_likelihood:
            pred_dist = self.mll.likelihood(self.mll.model(test_x))
        else:
            pred_dist = self.mll.model(test_x)
        return pred_dist

    def predict_batch(self, test_x, nn_size):
        """
        nn_size: number of nearest neighbors to use for prediction
        """
        test_nn_idx = self._compute_nn_idx(xd=self.train_x, xq=test_x, k=nn_size)

        mean = torch.zeros(
            test_x.shape[0], dtype=self.train_x.dtype, device=test_x.device
        )
        variance = torch.zeros(
            test_x.shape[0], dtype=self.train_x.dtype, device=test_x.device
        )
        for i in range(test_x.shape[0]):
            idx = test_nn_idx[i]
            pred_dist = self._predict(
                self.train_x[idx], self.train_y[idx], test_x[i : i + 1]
            )
            mean[i] = pred_dist.mean
            variance[i] = pred_dist.variance

        return torch.distributions.MultivariateNormal(mean, variance.diag())
