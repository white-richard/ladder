from ladder_api.config import ImageRepsConfig, TrainConfig


def test_resolve_paths_named_seed():
    cfg = TrainConfig(seed=3, checkpoints="out/RSNA/fold{seed}")
    resolved = cfg.resolve_paths()
    assert resolved.checkpoints == "out/RSNA/fold3"


def test_resolve_paths_positional_seed():
    cfg = ImageRepsConfig(seed=2, save_path="out/Waterbirds/seed{}")
    resolved = cfg.resolve_paths()
    assert resolved.save_path == "out/Waterbirds/seed2"


def test_to_namespace_fields():
    cfg = TrainConfig(weighted_BCE="y", balanced_dataloader="n")
    args = cfg.to_namespace()
    assert args.weighted_BCE == "y"
    assert args.balanced_dataloader == "n"
