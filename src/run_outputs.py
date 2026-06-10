import numpy as np
import pandas as pd
from tqdm import tqdm

from src.model import edit_model, generate, load_gemma
from settings import (
    DIRECTIONS_PATH,
    RANDOM_SEED,
    RUN_DIR,
    SPLIT_PATH,
)


MAX_NEW_TOKENS = 128
TEST_PROMPTS_PER_GROUP = None

BASELINE_OUTPUTS_PATH = RUN_DIR / "baseline_outputs.csv"
EDITED_OUTPUTS_PATH = RUN_DIR / "edited_outputs.csv"
EVALUATION_OUTPUTS_PATH = RUN_DIR / "evaluation_outputs.csv"

OUTPUT_FIELDS = ["id", "condition", "word_count", "answer"]


def main():
    if not SPLIT_PATH.exists() or not DIRECTIONS_PATH.exists():
        raise ValueError("run ablation/fit_direction.py first")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rows = test_rows(pd.read_csv(SPLIT_PATH))
    directions = np.load(DIRECTIONS_PATH, allow_pickle=False)["directions"]

    # Generate baseline outputs first, then edit the same loaded model.
    tokenizer, model, device = load_gemma()
    generate_condition(BASELINE_OUTPUTS_PATH, rows, "baseline", model, tokenizer, device)

    edit_model(model, directions)
    generate_condition(EDITED_OUTPUTS_PATH, rows, "edited", model, tokenizer, device)

    paired_rows(rows).to_csv(EVALUATION_OUTPUTS_PATH, index=False)
    print(f"wrote paired outputs: {EVALUATION_OUTPUTS_PATH}", flush=True)


def test_rows(rows):
    test = rows.loc[rows["split"] == "test"].copy()
    if TEST_PROMPTS_PER_GROUP is None:
        # Default project run uses every held-out test prompt.
        return test.sort_values(["test_group", "id"]).reset_index(drop=True)

    return (
        test
        .groupby("test_group", group_keys=False)
        .sample(n=TEST_PROMPTS_PER_GROUP, random_state=RANDOM_SEED)
        .sort_values(["test_group", "id"])
        .reset_index(drop=True)
    )


def generate_condition(path, rows, condition, model, tokenizer, device):
    done = completed_ids(path)
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
            "id": row.id,
            "condition": condition,
            "word_count": len(answer.split()),
            "answer": answer,
        }], columns=OUTPUT_FIELDS)
        output.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_ids(path):
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["id"])


def paired_rows(prompt_rows):
    baseline = pd.read_csv(BASELINE_OUTPUTS_PATH).rename(columns={
        "word_count": "baseline_word_count",
        "answer": "baseline_answer",
    })
    edited = pd.read_csv(EDITED_OUTPUTS_PATH).rename(columns={
        "word_count": "edited_word_count",
        "answer": "edited_answer",
    })

    paired = (
        prompt_rows
        .merge(baseline[["id", "baseline_word_count", "baseline_answer"]], on="id")
        .merge(edited[["id", "edited_word_count", "edited_answer"]], on="id")
    )
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


if __name__ == "__main__":
    main()
