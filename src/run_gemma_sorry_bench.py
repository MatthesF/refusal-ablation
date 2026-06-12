import numpy as np
import pandas as pd
from tqdm import tqdm

from settings import (
    DIRECTIONS_PATH,
    GENERATION_BATCH_SIZE,
    RUN_DIR,
    SORRY_BENCH_OUTPUTS_PATH,
)
from src.gemma import edit_gemma, generate_batch, load_gemma
from src.sorry_bench import current_run_id, load_questions, require_questions


OUTPUT_FIELDS = [
    "run_id",
    "question_id",
    "condition",
    "category",
    "max_new_tokens",
    "word_count",
    "answer_tokens",
    "hit_token_limit",
    "answer",
]


def main():
    require_questions()
    if not DIRECTIONS_PATH.exists():
        raise ValueError("run python -m src.fit_direction before SORRY-Bench generation")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    prepare_output_file()
    questions = load_questions()
    with np.load(DIRECTIONS_PATH, allow_pickle=False) as direction_file:
        if "split_style" not in direction_file.files:
            raise ValueError(
                "refusal_directions.npz is missing fit metadata; "
                "rerun python -m src.fit_direction"
            )
        directions = direction_file["directions"]
    run_id = current_run_id()

    tokenizer, model, device = load_gemma()
    generate_condition("baseline", questions, run_id, model, tokenizer, device)

    # The edit mutates Gemma in-place, so baseline generation must finish first.
    edit_gemma(model, directions)
    generate_condition("edited", questions, run_id, model, tokenizer, device)

    print(f"wrote Gemma SORRY-Bench outputs: {SORRY_BENCH_OUTPUTS_PATH}", flush=True)


def generate_condition(condition, questions, run_id, model, tokenizer, device):
    completed = completed_question_ids(condition, run_id)
    pending_questions = questions.loc[~questions["question_id"].isin(completed)]
    total_batches = (len(pending_questions) + GENERATION_BATCH_SIZE - 1) // GENERATION_BATCH_SIZE

    progress = tqdm(
        range(0, len(pending_questions), GENERATION_BATCH_SIZE),
        total=total_batches,
        desc=f"gemma {condition}",
        unit="batch",
    )
    for start in progress:
        batch = pending_questions.iloc[start:start + GENERATION_BATCH_SIZE]
        first_question = batch["question_id"].iloc[0]
        last_question = batch["question_id"].iloc[-1]
        progress.set_postfix(question_id=f"{first_question}-{last_question}")
        generations = generate_batch(
            model,
            tokenizer,
            batch["prompt"].tolist(),
            device,
        )
        output_rows = []
        for question, generation in zip(batch.itertuples(index=False), generations):
            answer, answer_tokens, max_new_tokens, hit_token_limit = generation
            output_rows.append({
                "run_id": run_id,
                "question_id": question.question_id,
                "condition": condition,
                "category": question.category,
                "max_new_tokens": max_new_tokens,
                "word_count": len(answer.split()),
                "answer_tokens": answer_tokens,
                "hit_token_limit": hit_token_limit,
                "answer": answer,
            })

        output = pd.DataFrame(output_rows, columns=OUTPUT_FIELDS)
        output.to_csv(
            SORRY_BENCH_OUTPUTS_PATH,
            mode="a",
            header=not SORRY_BENCH_OUTPUTS_PATH.exists(),
            index=False,
        )


def prepare_output_file():
    if not SORRY_BENCH_OUTPUTS_PATH.exists():
        return

    # The output file is append-only while a long GPU run is in flight. If the
    # schema changed between code versions, start clean rather than mixing rows.
    if list(pd.read_csv(SORRY_BENCH_OUTPUTS_PATH, nrows=0).columns) != OUTPUT_FIELDS:
        print(f"replacing output file with old columns: {SORRY_BENCH_OUTPUTS_PATH}", flush=True)
        SORRY_BENCH_OUTPUTS_PATH.unlink()


def completed_question_ids(condition, run_id):
    if not SORRY_BENCH_OUTPUTS_PATH.exists():
        return set()

    # Resume only the same run id and condition. A different direction file gets
    # a different run id, so previous answers cannot silently carry over.
    outputs = pd.read_csv(SORRY_BENCH_OUTPUTS_PATH)
    outputs = outputs.loc[
        (outputs["condition"] == condition)
        & (outputs["run_id"] == run_id)
    ]
    return set(outputs["question_id"])


if __name__ == "__main__":
    main()
