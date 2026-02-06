import torch
from torch import nn
from torch.utils.data import Dataset

from ladder_api import TrainConfig, Trainer, train_classifier


class DummyDataset(Dataset):
    def __init__(self, n: int = 16):
        self.x = torch.randn(n, 4)
        self.y = torch.randn(n, 1)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def test_train_classifier_runner_called():
    cfg = TrainConfig(
        seed=1,
        checkpoints="out/seed{seed}",
        output_path="out/seed{seed}",
        tensorboard_path="out/seed{seed}",
    )
    called = {}

    def runner(args):
        called["args"] = args

    artifacts = train_classifier(cfg, runner=runner, dry_run=False)
    assert "args" in called
    assert str(artifacts.checkpoints) == "out/seed1"


def test_train_classifier_dry_run():
    cfg = TrainConfig(seed=2, checkpoints="out/seed{seed}")
    artifacts = train_classifier(cfg, runner=lambda _: None, dry_run=True)
    assert str(artifacts.checkpoints) == "out/seed2"


def test_trainer_fit_cpu():
    dataset = DummyDataset(32)
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device="cpu")
    result = trainer.fit(dataset, epochs=2, batch_size=8)

    assert len(result.train_losses) == 2
    assert result.val_losses == []
