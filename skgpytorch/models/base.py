import torch
import gpytorch
import warnings


class BaseRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.history = {"train_loss": []}
        self.best_restart = None

    def fit(
        self,
        X=None,
        y=None,
        n_iters=1,
        n_restarts=1,
        verbose=0,
        verbose_gap=1,
        random_state=None,
    ):
        if random_state is not None:
            torch.manual_seed(random_state)

        if X is None:
            if hasattr(self.model, "train_inputs"):
                X = self.model.train_inputs[0]
            else:
                raise ValueError("X is required.")

        if y is None:
            if hasattr(self.model, "train_targets"):
                y = self.model.train_targets
            else:
                raise ValueError("y is required.")
        X = X.to(self.device)
        y = y.to(self.device)
        self.model.to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        if len(self.history["train_loss"]) > 0:
            warnings.warn(
                "Training loss is not empty. This may cause unexpected behavior."
            )

        best_loss = float("inf")
        for restart in range(n_restarts):
            self.history["train_loss"].append([])
            for param in self.model.parameters():
                torch.nn.init.normal_(param, mean=0.0, std=1.0)

            for i in range(n_iters):
                self.optimizer.zero_grad()
                loss = self(X, y)
                loss.backward()
                if verbose and i % verbose_gap == 0:
                    print(
                        "Restart: {}, Iter: {}, Loss: {:.4f}, Best Loss: {:.4f}".format(
                            restart, i, loss.item(), best_loss
                        )
                    )
                self.history["train_loss"][restart].append(loss.item())
                self.optimizer.step()

            # Last loss
            loss = self(X, y)

            # Check if best loss
            if loss.item() < best_loss:
                self.best_restart = restart
                best_loss = loss.item()
                best_model_state = self.model.state_dict()

            # Load the best model
            self.model.load_state_dict(best_model_state)
