from ladder_api import ErrorSliceConfig, LLMValidationConfig, discover_error_slices, validate_error_slices_llm


def test_discover_error_slices_calls_runner():
    cfg = ErrorSliceConfig(seed=1, save_path="out/seed{}")
    called = {}

    def runner(args):
        called["args"] = args

    artifacts = discover_error_slices(cfg, runner=runner, dry_run=False)
    assert "args" in called
    assert str(artifacts.save_path) == "out/seed1"
    assert str(artifacts.out_file).endswith("ladder_discover_slices_performance_ERM.txt")


def test_validate_error_slices_llm_calls_runner():
    cfg = LLMValidationConfig(seed=2, save_path="out/seed{}")
    called = {}

    def runner(args):
        called["args"] = args

    artifacts = validate_error_slices_llm(cfg, runner=runner, dry_run=False)
    assert "args" in called
    assert str(artifacts.save_path) == "out/seed2"
