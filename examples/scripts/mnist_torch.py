import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision.transforms import ToTensor
from tqdm import tqdm

import approx

approx.auto_select_backend()


class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 32 * 7 * 7)
        logits = self.classifier(x)
        return logits


mnist_train = torchvision.datasets.MNIST(
    root="./datasets", train=True, download=True, transform=ToTensor()
)
mnist_test = torchvision.datasets.MNIST(
    root="./datasets", train=False, download=True, transform=ToTensor()
)

train_loader = data.DataLoader(mnist_train, batch_size=512, shuffle=True)
test_loader = data.DataLoader(mnist_test, batch_size=512, shuffle=True)

model = BasicNN()
model.to("cuda")
optim = torch.optim.SGD(model.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss().to("cuda")

with approx.auto_cast_all():
    num_epochs = 10
    for epoch_it in range(num_epochs):
        prog = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_num, (features, labels) in prog:
            features = features.to("cuda")
            labels = labels.to("cuda")
            logits = model(features)
            loss_value = loss(logits, labels)
            optim.zero_grad()
            loss_value.backward()
            optim.step()

            prog.set_postfix(
                {"epoch": epoch_it + 1, "loss": loss_value.item()}
            )
        prog.close()
