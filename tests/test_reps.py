from ladder_api import ImageRepsConfig, TextRepsConfig, save_image_reps, save_text_reps


def test_save_image_reps_calls_runner():
    cfg = ImageRepsConfig(
        seed=1,
        save_path="out/seed{}",
        clip_vision_encoder="ViT-B/32",
    )
    called = {}

    def runner(args):
        called["args"] = args

    artifacts = save_image_reps(cfg, runner=runner, dry_run=False)
    assert "args" in called
    assert str(artifacts.save_path) == "out/seed1/clip_img_encoder_ViT-B/32"


def test_save_text_reps_calls_runner():
    cfg = TextRepsConfig(
        seed=2,
        save_path="out/seed{}",
        clip_vision_encoder="RN50",
    )
    called = {}

    def runner(args):
        called["args"] = args

    artifacts = save_text_reps(cfg, runner=runner, dry_run=False)
    assert "args" in called
    assert str(artifacts.save_path) == "out/seed2/clip_img_encoder_RN50"
