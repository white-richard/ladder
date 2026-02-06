from ladder_api import (
    AlignerConfig,
    ErrorSliceConfig,
    ImageRepsConfig,
    LLMValidationConfig,
    LadderPipeline,
    MitigationConfig,
    PipelineConfig,
    TextRepsConfig,
    TrainConfig,
)


def test_pipeline_run_sequence():
    calls = []

    def make_runner(name):
        def _runner(args):
            calls.append(name)
        return _runner

    cfg = PipelineConfig(
        train=TrainConfig(seed=1, checkpoints="out/seed{seed}"),
        image_reps=ImageRepsConfig(seed=1, save_path="out/seed{}"),
        text_reps=TextRepsConfig(seed=1, save_path="out/seed{}"),
        aligner=AlignerConfig(seed=1, save_path="out/seed{}/aligner"),
        error_slices=ErrorSliceConfig(seed=1, save_path="out/seed{}"),
        llm_validation=LLMValidationConfig(seed=1, save_path="out/seed{}"),
        mitigation=MitigationConfig(seed=1, save_path="out/seed{}"),
    )

    pipeline = LadderPipeline(cfg)
    artifacts = pipeline.run(
        dry_run=False,
        runners={
            "train": make_runner("train"),
            "image_reps": make_runner("image_reps"),
            "text_reps": make_runner("text_reps"),
            "aligner": make_runner("aligner"),
            "error_slices": make_runner("error_slices"),
            "llm_validation": make_runner("llm_validation"),
            "mitigation": make_runner("mitigation"),
        },
    )

    assert calls == [
        "train",
        "image_reps",
        "text_reps",
        "aligner",
        "error_slices",
        "llm_validation",
        "mitigation",
    ]
    assert artifacts.train is not None
    assert artifacts.mitigation is not None
