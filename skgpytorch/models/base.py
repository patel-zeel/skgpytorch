import torch
import gpytorch
import warnings
import faiss
import numpy as np


class BaseRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.history = {"train_loss": []}
        self.best_restart = None
        self.res = faiss.StandardGpuResources()

    def compute_train_nn_idx(self, k):
        x = (self.train_x.data.float()).cpu().numpy()
        self.cpu_index = faiss.IndexFlatL2(self.train_x.size(-1))
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 1, self.cpu_index)
        self.gpu_index.add(x)
        self.train_nn_idx = (
            torch.from_numpy(self.gpu_index.search(x, k)[1])
            .long()
            .to(self.train_x.device)
        )

    def fit(
        self,
        batch_size=None,
        lr=0.1,
        n_epochs=1,
        n_restarts=1,
        verbose=0,
        verbose_gap=1,
        random_state=None,
    ):
        if random_state is not None:
            torch.manual_seed(random_state)
        if batch_size is None:
            batch_size = self.train_x.shape[0]

        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if len(self.history["train_loss"]) > 0:
            warnings.warn(
                "Training loss is not empty. This may cause unexpected behavior."
            )

        self.compute_train_nn_idx(k=batch_size)

        best_loss = float("inf")
        for restart in range(n_restarts):
            self.history["train_loss"].append([])
            for param in self.model.parameters():
                torch.nn.init.normal_(param, mean=0.0, std=1.0)

            for epoch in range(n_epochs):
                for iteration in range(self.train_x.shape[0] // batch_size):
                    idx = torch.randint(low=0, high=self.train_x.shape[0], size=(1,))[0]
                    indices = self.train_nn_idx[idx]
                    X_batch = self.train_x[indices]
                    y_batch = self.train_y[indices]
                    # X_batch = self.train_x
                    # y_batch = self.train_y

                    self.optimizer.zero_grad()
                    loss = self(X_batch, y_batch)
                    loss.backward()
                    if verbose and epoch % verbose_gap == 0:
                        print(
                            "Restart: {}, Iter: {}, Loss: {:.4f}, Best Loss: {:.4f}".format(
                                restart, epoch, loss.item(), best_loss
                            )
                        )
                    self.history["train_loss"][restart].append(loss.item())
                    self.optimizer.step()

            # Last loss
            # This can consume entire RAM
            loss = self(self.train_x, self.train_y)

            # Check if best loss
            if loss.item() < best_loss:
                self.best_restart = restart
                best_loss = loss.item()
                best_model_state = self.model.state_dict()

            # Load the best model
            self.model.load_state_dict(best_model_state)
