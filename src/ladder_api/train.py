"""Training helpers and a lightweight trainer for custom datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from .compat import ensure_codebase_on_path
from .config import TrainConfig


@dataclass
class TrainArtifacts:
    """Resolved output locations for a training run."""

    checkpoints: Path
    output_path: Path
    tensorboard_path: Path


@dataclass
class TrainingResult:
    """Loss history for a training run."""

    train_losses: List[float]
    val_losses: List[float]


def _resolve_artifacts(args) -> TrainArtifacts:
    return TrainArtifacts(
        checkpoints=Path(args.checkpoints),
        output_path=Path(args.output_path),
        tensorboard_path=Path(args.tensorboard_path),
    )


def train_classifier(
    config: TrainConfig,
    runner: Optional[Callable[[object], None]] = None,
    dry_run: bool = False,
) -> TrainArtifacts:
    """Run the legacy RSNA/VinDr classifier training entrypoint.

    Args:
        config: Train configuration.
        runner: Optional callable to run instead of the legacy `main`.
        dry_run: If True, skip execution and only return resolved artifacts.

    Returns:
        TrainArtifacts with resolved output paths.
    """
    ensure_codebase_on_path()
    args = config.to_namespace()
    artifacts = _resolve_artifacts(args)
    if dry_run:
        return artifacts

    if runner is None:
        # Import lazily to avoid heavy module import when not needed.
        from train_classifier_Mammo import main as runner

    runner(args)
    return artifacts


class Trainer:
    """Simple training loop for custom datasets or dataloaders."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def _ensure_loader(
        self,
        data: Union[DataLoader, Dataset],
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ) -> DataLoader:
        if isinstance(data, DataLoader):
            return data
        if isinstance(data, Dataset):
            return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        raise TypeError("Expected torch Dataset or DataLoader")

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            if "x" not in batch or "y" not in batch:
                raise KeyError("Batch dict must contain 'x' and 'y' keys")
            return batch["x"], batch["y"]
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        raise TypeError("Batch must be a dict with x/y or a (x, y) tuple")

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in loader:
            inputs, targets = self._unpack_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_batches += 1
        return total_loss / max(1, total_batches)

    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in loader:
                inputs, targets = self._unpack_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += float(loss.detach().cpu())
                total_batches += 1
        return total_loss / max(1, total_batches)

    def fit(
        self,
        train_data: Union[DataLoader, Dataset],
        val_data: Optional[Union[DataLoader, Dataset]] = None,
        epochs: int = 1,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> TrainingResult:
        """Train the model for a number of epochs.

        Args:
            train_data: Dataset or DataLoader for training.
            val_data: Optional Dataset or DataLoader for validation.
            epochs: Number of epochs.
            batch_size: Batch size if a Dataset is provided.
            shuffle: Whether to shuffle training data when Dataset is provided.
            num_workers: DataLoader workers if Dataset is provided.
            scheduler: Optional LR scheduler.

        Returns:
            TrainingResult with train and validation losses.
        """
        train_loader = self._ensure_loader(train_data, batch_size, shuffle, num_workers)
        val_loader = None
        if val_data is not None:
            val_loader = self._ensure_loader(val_data, batch_size, False, num_workers)

        train_losses: List[float] = []
        val_losses: List[float] = []

        for _ in range(epochs):
            train_losses.append(self._train_epoch(train_loader))
            if val_loader is not None:
                val_losses.append(self._eval_epoch(val_loader))
            if scheduler is not None:
                scheduler.step()

        return TrainingResult(train_losses=train_losses, val_losses=val_losses)
