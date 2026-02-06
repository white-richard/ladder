from ladder_api import AlignerConfig, train_aligner


def test_train_aligner_calls_runner():
    cfg = AlignerConfig(seed=3, save_path="out/seed{}/aligner")
    called = {}

    def runner(args):
        called["args"] = args

    artifacts = train_aligner(cfg, runner=runner, dry_run=False)
    assert "args" in called
    assert str(artifacts.save_path) == "out/seed3/aligner"
