"""Representation saving helpers for images and text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .compat import ensure_codebase_on_path
from .config import ImageRepsConfig, TextRepsConfig


@dataclass
class RepsArtifacts:
    """Resolved output locations for representation saving."""

    save_path: Path


def _final_save_path(base_path: Path, clip_vision_encoder: str) -> Path:
    return base_path / f"clip_img_encoder_{clip_vision_encoder}"


def save_image_reps(
    config: ImageRepsConfig,
    runner: Optional[Callable[[object], None]] = None,
    dry_run: bool = False,
) -> RepsArtifacts:
    """Generate and save image representations using legacy logic."""
    ensure_codebase_on_path()
    args = config.to_namespace()
    base = Path(args.save_path)
    final = _final_save_path(base, args.clip_vision_encoder)
    artifacts = RepsArtifacts(save_path=final)
    if dry_run:
        return artifacts

    if runner is None:
        from save_img_reps import main as runner

    runner(args)
    return artifacts


def save_text_reps(
    config: TextRepsConfig,
    runner: Optional[Callable[[object], None]] = None,
    dry_run: bool = False,
) -> RepsArtifacts:
    """Generate and save text representations using legacy logic."""
    ensure_codebase_on_path()
    args = config.to_namespace()
    base = Path(args.save_path)
    final = _final_save_path(base, args.clip_vision_encoder)
    artifacts = RepsArtifacts(save_path=final)
    if dry_run:
        return artifacts

    if runner is None:
        from save_text_reps import main as runner

    runner(args)
    return artifacts
