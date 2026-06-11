import os

from settings import (
    SORRY_BENCH_DATASET_ID,
    SORRY_BENCH_DATASET_REVISION,
    SORRY_BENCH_DIR,
    SORRY_BENCH_JUDGE_LOCAL_DIR,
    SORRY_BENCH_JUDGE_MODEL_ID,
    SORRY_BENCH_JUDGE_MODEL_REVISION,
)
from src.sorry_bench import download_official_evaluator, require_questions


def main():
    download_snapshot(
        repo_id=SORRY_BENCH_DATASET_ID,
        repo_type="dataset",
        revision=SORRY_BENCH_DATASET_REVISION,
        local_dir=SORRY_BENCH_DIR,
    )
    require_questions()
    print(f"downloaded SORRY-Bench dataset: {SORRY_BENCH_DIR}", flush=True)

    download_snapshot(
        repo_id=SORRY_BENCH_JUDGE_MODEL_ID,
        repo_type="model",
        revision=SORRY_BENCH_JUDGE_MODEL_REVISION,
        local_dir=SORRY_BENCH_JUDGE_LOCAL_DIR,
    )
    print(f"downloaded SORRY-Bench judge model: {SORRY_BENCH_JUDGE_LOCAL_DIR}", flush=True)

    download_official_evaluator()
    print("downloaded pinned official SORRY-Bench evaluator", flush=True)


def download_snapshot(repo_id, repo_type, revision, local_dir):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("install requirements.txt before downloading assets") from error

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
            f"failed to download {repo_type} '{repo_id}'. "
            "For SORRY-Bench, accept the Hugging Face terms and export HF_TOKEN."
        ) from error


if __name__ == "__main__":
    main()
