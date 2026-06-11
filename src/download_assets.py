from settings import (
    SORRY_BENCH_DATASET_ID,
    SORRY_BENCH_DATASET_REVISION,
    SORRY_BENCH_DIR,
)
from src.huggingface_download import download_snapshot
from src.sorry_bench import download_official_evaluator, require_questions


def main():
    download_snapshot(
        repo_id=SORRY_BENCH_DATASET_ID,
        repo_type="dataset",
        revision=SORRY_BENCH_DATASET_REVISION,
        local_dir=SORRY_BENCH_DIR,
        access_note=(
            "Accept the SORRY-Bench dataset terms and export HF_TOKEN if "
            "Hugging Face requires authentication."
        ),
    )
    require_questions()
    print(f"downloaded SORRY-Bench dataset: {SORRY_BENCH_DIR}", flush=True)

    download_official_evaluator()
    print("downloaded pinned official SORRY-Bench evaluator", flush=True)


if __name__ == "__main__":
    main()
