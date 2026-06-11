from settings import (
    SORRY_BENCH_JUDGE_LOCAL_DIR,
    SORRY_BENCH_JUDGE_MODEL_ID,
    SORRY_BENCH_JUDGE_MODEL_REVISION,
)
from src.huggingface_download import download_snapshot


def main():
    download_snapshot(
        repo_id=SORRY_BENCH_JUDGE_MODEL_ID,
        repo_type="model",
        revision=SORRY_BENCH_JUDGE_MODEL_REVISION,
        local_dir=SORRY_BENCH_JUDGE_LOCAL_DIR,
        access_note=(
            "This is the gated official SORRY-Bench judge checkpoint. "
            "Request access on Hugging Face, wait for approval, then export HF_TOKEN."
        ),
    )
    print(f"downloaded SORRY-Bench judge model: {SORRY_BENCH_JUDGE_LOCAL_DIR}", flush=True)


if __name__ == "__main__":
    main()
