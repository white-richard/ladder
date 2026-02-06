"""Error slice mitigation wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .compat import ensure_codebase_on_path
from .config import MitigationConfig


@dataclass
class MitigationArtifacts:
    """Resolved output locations for mitigation."""

    save_path: Path
    out_file: Path


def _format_seed_template(value: str, seed: int) -> str:
    if "{seed}" in value:
        return value.format(seed=seed)
    try:
        return value.format(seed)
    except Exception:
        return value


def mitigate_error_slices(
    config: MitigationConfig,
    runner: Optional[Callable[[object], None]] = None,
    dry_run: bool = False,
) -> MitigationArtifacts:
    """Mitigate error slices using legacy logic."""
    ensure_codebase_on_path()
    args = config.to_namespace()
    save_path = Path(_format_seed_template(str(args.save_path), args.seed))
    out_file = save_path / "ladder_mitigate_slices.txt"
    artifacts = MitigationArtifacts(save_path=save_path, out_file=out_file)
    if dry_run:
        return artifacts

    if runner is None:
        from mitigate_error_slices import main as runner

    runner(args)
    return artifacts
