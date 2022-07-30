from typing import List

import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

import approx

train_ds = data.TensorDataset(
    torch.tensor([float(x) for x in range(100)]),
    torch.tensor([float(5 + 3 * x) for x in range(100)]),
)
test_ds = data.TensorDataset(
    torch.tensor([float(x) for x in range(101, 131)]),
    torch.tensor([float(5 + 3 * x) for x in range(101, 131)]),
)
train_dl = data.DataLoader(train_ds, batch_size=8, shuffle=True)
test_dl = data.DataLoader(test_ds, batch_size=8, shuffle=True)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))

    def forward(self, x):
        return self.net(x)


def train_mdl(
    train_data: data.DataLoader,
    model: ToyModel,
    optim: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs=10,
):
    model.train()
    for epoch in range(epochs):
        prog = tqdm(enumerate(train_data), total=len(train_data))
        for b, (x, y) in prog:
            optim.zero_grad()
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optim.step()
            prog.set_postfix({"epoch": epoch, "loss": loss.item()})
        prog.close()


def eval_loop(model: ToyModel) -> List[List[float]]:
    model.eval()
    loss_fn = nn.MSELoss()
    loss_history = []
    with torch.no_grad():
        prog = tqdm(enumerate(test_dl), total=len(test_dl))
        for b, (x, y) in prog:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss_history.append(loss.item())
            prog.set_postfix({"loss": loss.item()})
        prog.close()
    return [loss_history]


def main():
    model = ToyModel()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    train_mdl(train_dl, model, optim, loss_fn, epochs=10)
    result = approx.compare(model, model, eval_loop=eval_loop)
    print(result)


if __name__ == "__main__":
    main()
