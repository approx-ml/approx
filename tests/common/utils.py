import functools
from typing import Callable

import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm


def func_fixture(func: Callable):
    """
    A decorator to make a fixture that returns a function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


class SimpleRandomModelPack:
    def __init__(self):
        self.model = _SimpleModel()
        self.train_ds = data.TensorDataset(
            torch.tensor([float(x) for x in range(100)]),
            torch.tensor([float(5 + 3 * x) for x in range(100)]),
        )
        self.test_ds = data.TensorDataset(
            torch.tensor([float(x) for x in range(101, 131)]),
            torch.tensor([float(5 + 3 * x) for x in range(101, 131)]),
        )
        self.test_dl = data.DataLoader(
            self.test_ds, batch_size=32, shuffle=True
        )
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.epochs = 10

    def train_model(self):
        self.model.train()
        train_data = data.DataLoader(
            self.train_ds, batch_size=32, shuffle=True
        )
        for epoch in range(self.epochs):
            prog = tqdm(
                enumerate(train_data), total=len(train_data), disable=True
            )
            for b, (x, y) in prog:
                self.optim.zero_grad()
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optim.step()
                prog.set_postfix({"epoch": epoch, "loss": loss.item()})
            prog.close()

    # noinspection DuplicatedCode
    @staticmethod
    def eval_loop(model: _SimpleModel, test_dl: data.DataLoader):
        model.eval()
        loss_fn = nn.MSELoss()
        loss_history = []
        acc_history = []
        with torch.no_grad():
            prog = tqdm(enumerate(test_dl), total=len(test_dl), disable=True)
            for b, (x, y) in prog:
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                acc = (
                    torch.where(torch.abs(y_pred.round() - y) < 0.5, 1.0, 0.0)
                    .float()
                    .mean()
                )
                loss_history.append(loss.item())
                acc_history.append(acc.item())
                prog.set_postfix({"loss": loss.item()})
            prog.close()

        return {"loss": loss_history, "accuracy": acc_history}
