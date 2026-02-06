"""Example: Use the generic Trainer with a custom Dataset."""

from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

from ladder_api import Trainer


class ToyDataset(Dataset):
    def __init__(self, n: int = 128) -> None:
        self.x = torch.randn(n, 10)
        self.y = (self.x.sum(dim=1, keepdim=True) > 0).float()

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def main() -> None:
    train_ds = ToyDataset(256)
    val_ds = ToyDataset(64)

    model = nn.Sequential(nn.Linear(10, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device="cpu")
    result = trainer.fit(train_ds, val_ds, epochs=3, batch_size=32)
    print(result)


if __name__ == "__main__":
    main()
