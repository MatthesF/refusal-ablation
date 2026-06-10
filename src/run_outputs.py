import hashlib

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.model import edit_model, generate, load_gemma
from settings import (
    DIRECTIONS_PATH,
    MAX_NEW_TOKENS,
    RUN_DIR,
    SPLIT_PATH,
)


BASELINE_OUTPUTS_PATH = RUN_DIR / "baseline_outputs.csv"
EDITED_OUTPUTS_PATH = RUN_DIR / "edited_outputs.csv"
EVALUATION_OUTPUTS_PATH = RUN_DIR / "evaluation_outputs.csv"

OUTPUT_FIELDS = ["run_id", "id", "condition", "max_new_tokens", "word_count", "answer"]


def main():
    if not SPLIT_PATH.exists() or not DIRECTIONS_PATH.exists():
        raise ValueError("run python -m src.fit_direction first")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rows = evaluation_rows(pd.read_csv(SPLIT_PATH))
    directions = load_directions()
    run_id = output_run_id()

    # Same loaded model, same prompts: first baseline, then edited.
    tokenizer, model, device = load_gemma()
    generate_condition(BASELINE_OUTPUTS_PATH, rows, "baseline", run_id, model, tokenizer, device)

    edit_model(model, directions)
    generate_condition(EDITED_OUTPUTS_PATH, rows, "edited", run_id, model, tokenizer, device)

    paired_rows(rows, run_id).to_csv(EVALUATION_OUTPUTS_PATH, index=False)
    print(f"wrote paired outputs: {EVALUATION_OUTPUTS_PATH}", flush=True)


def load_directions():
    with np.load(DIRECTIONS_PATH, allow_pickle=False) as direction_file:
        if "split_style" not in direction_file.files:
            raise ValueError("directions were made by the old pipeline; run python -m src.fit_direction")
        return direction_file["directions"]


def output_run_id():
    # Changing the fitted direction or token limit creates a new output run.
    direction_hash = hashlib.sha256(DIRECTIONS_PATH.read_bytes()).hexdigest()[:12]
    return f"{direction_hash}-tok{MAX_NEW_TOKENS}"


def evaluation_rows(rows):
    return (
        rows.loc[rows["split"] == "test"]
        .sort_values(["test_group", "id"])
        .reset_index(drop=True)
    )


def generate_condition(path, rows, condition, run_id, model, tokenizer, device):
    prepare_output_file(path)
    done = completed_ids(path, condition, run_id)
    rows_to_run = rows.loc[~rows["id"].isin(done)]

    # Append one row at a time so an interrupted long run can resume.
    progress = tqdm(
        rows_to_run.itertuples(index=False),
        total=len(rows_to_run),
        desc=condition,
        unit="prompt",
    )
    for row in progress:
        progress.set_postfix(id=row.id)
        answer = generate(model, tokenizer, row.prompt, device, MAX_NEW_TOKENS)
        output = pd.DataFrame([{
            "run_id": run_id,
            "id": row.id,
            "condition": condition,
            "max_new_tokens": MAX_NEW_TOKENS,
            "word_count": len(answer.split()),
            "answer": answer,
        }], columns=OUTPUT_FIELDS)
        output.to_csv(path, mode="a", header=not path.exists(), index=False)


def prepare_output_file(path):
    if not path.exists():
        return
    columns = list(pd.read_csv(path, nrows=0).columns)
    if columns != OUTPUT_FIELDS:
        # These are generated checkpoints; manual labels are written to the paired file.
        print(f"replacing stale output file: {path}", flush=True)
        path.unlink()


def completed_ids(path, condition, run_id):
    if not path.exists():
        return set()
    outputs = pd.read_csv(path)
    outputs = outputs.loc[
        (outputs["condition"] == condition)
        & (outputs["run_id"] == run_id)
        & (outputs["max_new_tokens"] == MAX_NEW_TOKENS)
    ]
    return set(outputs["id"])


def paired_rows(prompt_rows, run_id):
    # Manual labeling is easiest from one row per prompt with both model answers.
    baseline = current_outputs(BASELINE_OUTPUTS_PATH, "baseline", run_id).rename(columns={
        "word_count": "baseline_word_count",
        "answer": "baseline_answer",
    })
    edited = current_outputs(EDITED_OUTPUTS_PATH, "edited", run_id).rename(columns={
        "word_count": "edited_word_count",
        "answer": "edited_answer",
    })

    paired = (
        prompt_rows
        .merge(baseline[["id", "baseline_word_count", "baseline_answer"]], on="id")
        .merge(edited[["id", "edited_word_count", "edited_answer"]], on="id")
    )
    if len(paired) != len(prompt_rows):
        raise ValueError("not all test prompts have both baseline and edited outputs")

    paired["baseline_manual_label"] = ""
    paired["edited_manual_label"] = ""

    # These blank columns are filled manually before the statistical analysis.
    return paired[[
        "id",
        "label",
        "category",
        "test_group",
        "prompt",
        "baseline_manual_label",
        "edited_manual_label",
        "baseline_word_count",
        "edited_word_count",
        "baseline_answer",
        "edited_answer",
    ]]


def current_outputs(path, condition, run_id):
    outputs = pd.read_csv(path)
    return outputs.loc[
        (outputs["condition"] == condition)
        & (outputs["run_id"] == run_id)
        & (outputs["max_new_tokens"] == MAX_NEW_TOKENS)
    ]


if __name__ == "__main__":
    main()
