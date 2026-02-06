from ladder_api import MitigationConfig, mitigate_error_slices


def test_mitigate_error_slices_calls_runner():
    cfg = MitigationConfig(seed=4, save_path="out/seed{}")
    called = {}

    def runner(args):
        called["args"] = args

    artifacts = mitigate_error_slices(cfg, runner=runner, dry_run=False)
    assert "args" in called
    assert str(artifacts.save_path) == "out/seed4"
    assert str(artifacts.out_file).endswith("ladder_mitigate_slices.txt")
