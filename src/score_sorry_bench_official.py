from settings import (
    SORRY_BENCH_OFFICIAL_JUDGMENTS_PATH,
    SORRY_BENCH_OFFICIAL_SUMMARY_PATH,
    SORRY_BENCH_OUTPUTS_PATH,
)
from src.sorry_bench import export_official_answers, import_official_judgments, run_official_judge


def main():
    if not SORRY_BENCH_OUTPUTS_PATH.exists():
        raise ValueError("run python -m src.run_gemma_sorry_bench before official scoring")

    manifest = export_official_answers()
    print("exported official answer files", flush=True)

    command = run_official_judge(manifest)
    print("ran official SORRY-Bench judge:", " ".join(command), flush=True)

    judgments, summary = import_official_judgments(manifest)
    print(
        f"wrote official judgments: {SORRY_BENCH_OFFICIAL_JUDGMENTS_PATH} "
        f"({len(judgments)} rows)",
        flush=True,
    )
    print(
        f"wrote official summary: {SORRY_BENCH_OFFICIAL_SUMMARY_PATH} "
        f"({len(summary)} rows)",
        flush=True,
    )


if __name__ == "__main__":
    main()
