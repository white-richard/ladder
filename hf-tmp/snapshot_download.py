from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="shawn24/Ladder",
    allow_patterns="out/*",
    local_dir=".hf-tmp/Ladder_data"
)