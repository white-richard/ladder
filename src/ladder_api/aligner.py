"""Aligner training wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .compat import ensure_codebase_on_path
from .config import AlignerConfig


@dataclass
class AlignerArtifacts:
    """Resolved output locations for aligner training."""

    save_path: Path


def _format_seed_template(value: str, seed: int) -> str:
    if "{seed}" in value:
        return value.format(seed=seed)
    try:
        return value.format(seed)
    except Exception:
        return value


def train_aligner(
    config: AlignerConfig,
    runner: Optional[Callable[[object], None]] = None,
    dry_run: bool = False,
) -> AlignerArtifacts:
    """Train the linear aligner using legacy logic."""
    ensure_codebase_on_path()
    args = config.to_namespace()
    save_path = Path(_format_seed_template(str(args.save_path), args.seed))
    artifacts = AlignerArtifacts(save_path=save_path)
    if dry_run:
        return artifacts

    if runner is None:
        from learn_aligner import main as runner

    runner(args)
    return artifacts
