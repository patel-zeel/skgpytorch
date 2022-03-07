from abc import abstractmethod


class CallBack:
    @abstractmethod
    def __call__(self, locals):
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        raise NotImplementedError


class History(CallBack):
    def __init__(self):
        self.history = []

    def __call__(self, locals):
        self.history.append(locals["batch_loss"].item())

    def finalize(self):
        pass
