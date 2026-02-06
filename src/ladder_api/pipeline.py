"""High-level pipeline runner for the LADDER workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .aligner import AlignerArtifacts, train_aligner
from .config import (
    AlignerConfig,
    ErrorSliceConfig,
    ImageRepsConfig,
    LLMValidationConfig,
    MitigationConfig,
    TextRepsConfig,
    TrainConfig,
)
from .error_slices import ErrorSliceArtifacts, LLMValidationArtifacts, discover_error_slices, validate_error_slices_llm
from .mitigation import MitigationArtifacts, mitigate_error_slices
from .reps import RepsArtifacts, save_image_reps, save_text_reps
from .train import TrainArtifacts, train_classifier


@dataclass
class PipelineConfig:
    """Configuration bundle for the full LADDER pipeline."""

    train: Optional[TrainConfig] = None
    image_reps: Optional[ImageRepsConfig] = None
    text_reps: Optional[TextRepsConfig] = None
    aligner: Optional[AlignerConfig] = None
    error_slices: Optional[ErrorSliceConfig] = None
    llm_validation: Optional[LLMValidationConfig] = None
    mitigation: Optional[MitigationConfig] = None

    run_train: bool = True
    run_image_reps: bool = True
    run_text_reps: bool = True
    run_aligner: bool = True
    run_error_slices: bool = True
    run_llm_validation: bool = True
    run_mitigation: bool = True


@dataclass
class PipelineArtifacts:
    """Artifacts produced by the pipeline, if any steps were run."""

    train: Optional[TrainArtifacts] = None
    image_reps: Optional[RepsArtifacts] = None
    text_reps: Optional[RepsArtifacts] = None
    aligner: Optional[AlignerArtifacts] = None
    error_slices: Optional[ErrorSliceArtifacts] = None
    llm_validation: Optional[LLMValidationArtifacts] = None
    mitigation: Optional[MitigationArtifacts] = None


class LadderPipeline:
    """Execute the LADDER pipeline steps in order."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(
        self,
        dry_run: bool = False,
        runners: Optional[Dict[str, Callable[[object], None]]] = None,
    ) -> PipelineArtifacts:
        runners = runners or {}
        artifacts = PipelineArtifacts()

        if self.config.run_train and self.config.train is not None:
            artifacts.train = train_classifier(
                self.config.train, runner=runners.get("train"), dry_run=dry_run
            )

        if self.config.run_image_reps and self.config.image_reps is not None:
            artifacts.image_reps = save_image_reps(
                self.config.image_reps, runner=runners.get("image_reps"), dry_run=dry_run
            )

        if self.config.run_text_reps and self.config.text_reps is not None:
            artifacts.text_reps = save_text_reps(
                self.config.text_reps, runner=runners.get("text_reps"), dry_run=dry_run
            )

        if self.config.run_aligner and self.config.aligner is not None:
            artifacts.aligner = train_aligner(
                self.config.aligner, runner=runners.get("aligner"), dry_run=dry_run
            )

        if self.config.run_error_slices and self.config.error_slices is not None:
            artifacts.error_slices = discover_error_slices(
                self.config.error_slices, runner=runners.get("error_slices"), dry_run=dry_run
            )

        if self.config.run_llm_validation and self.config.llm_validation is not None:
            artifacts.llm_validation = validate_error_slices_llm(
                self.config.llm_validation, runner=runners.get("llm_validation"), dry_run=dry_run
            )

        if self.config.run_mitigation and self.config.mitigation is not None:
            artifacts.mitigation = mitigate_error_slices(
                self.config.mitigation, runner=runners.get("mitigation"), dry_run=dry_run
            )

        return artifacts
