import os


def download_snapshot(repo_id, repo_type, revision, local_dir, access_note):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "install requirements.txt before downloading Hugging Face assets"
        ) from error

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=str(local_dir),
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as error:
        raise RuntimeError(
            f"failed to download {repo_type} '{repo_id}'. {access_note}"
        ) from error
