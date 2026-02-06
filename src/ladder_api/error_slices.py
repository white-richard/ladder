"""Error slice discovery and validation wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .compat import ensure_codebase_on_path
from .config import ErrorSliceConfig, LLMValidationConfig


@dataclass
class ErrorSliceArtifacts:
    """Resolved output locations for error slice discovery."""

    save_path: Path
    out_file: Path


@dataclass
class LLMValidationArtifacts:
    """Resolved output locations for LLM validation."""

    save_path: Path


def _format_seed_template(value: str, seed: int) -> str:
    if "{seed}" in value:
        return value.format(seed=seed)
    try:
        return value.format(seed)
    except Exception:
        return value


def discover_error_slices(
    config: ErrorSliceConfig,
    runner: Optional[Callable[[object], None]] = None,
    dry_run: bool = False,
) -> ErrorSliceArtifacts:
    """Discover error slices using legacy logic."""
    ensure_codebase_on_path()
    args = config.to_namespace()
    save_path = Path(_format_seed_template(str(args.save_path), args.seed))
    out_file = save_path / "ladder_discover_slices_performance_ERM.txt"
    artifacts = ErrorSliceArtifacts(save_path=save_path, out_file=out_file)
    if dry_run:
        return artifacts

    if runner is None:
        from discover_error_slices import main as runner

    runner(args)
    return artifacts


def validate_error_slices_llm(
    config: LLMValidationConfig,
    runner: Optional[Callable[[object], None]] = None,
    dry_run: bool = False,
) -> LLMValidationArtifacts:
    """Validate error slices via LLM hypotheses using legacy logic."""
    ensure_codebase_on_path()
    args = config.to_namespace()
    save_path = Path(_format_seed_template(str(args.save_path), args.seed))
    artifacts = LLMValidationArtifacts(save_path=save_path)
    if dry_run:
        return artifacts

    if runner is None:
        from validate_error_slices_w_LLM import main as runner

    runner(args)
    return artifacts
