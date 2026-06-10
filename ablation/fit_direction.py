import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from dataset import load_prompts, split_prompts
from model import last_prompt_token_activations, load_gemma, refusal_directions
from settings import (
    CONSTRUCTION_ACTIVATIONS_PATH,
    DIRECTIONS_PATH,
    HELD_OUT_UNSAFE_CATEGORIES,
    MODEL_ID,
    MODEL_REVISION,
    RANDOM_SEED,
    RUN_DIR,
    SPLIT_PATH,
)


PROMPTS_PER_LABEL = 50
KFOLD_SPLITS = 5

FIT_REPORT_PATH = RUN_DIR / "direction_fit_report.csv"


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # The split is written once and reused by the output script.
    split_rows = split_prompts(load_prompts())
    split_rows.to_csv(SPLIT_PATH, index=False)

    construction_rows = split_rows.loc[split_rows["split"] == "construction"].reset_index(drop=True)
    activations = load_or_make_activations(construction_rows)

    # The direction is fitted from a fixed, category-balanced 50 safe + 50 unsafe prompts.
    fit_rows = choose_fit_rows(construction_rows)
    fit_labels = fit_rows["label"].to_numpy()
    fit_activations = activations[fit_rows.index.to_numpy()]

    # K-fold is reported as uncertainty/stability evidence; it is not used as a pass/fail rule.
    report = kfold_report(fit_activations, fit_labels)
    pd.DataFrame([report]).to_csv(FIT_REPORT_PATH, index=False)

    # The final edit uses the full fixed 100-prompt fit set.
    directions = refusal_directions(fit_activations, fit_labels)
    np.savez_compressed(
        DIRECTIONS_PATH,
        directions=directions,
        model_id=np.array([MODEL_ID]),
        model_revision=np.array([MODEL_REVISION]),
        fit_prompt_ids=fit_rows["id"].to_numpy(),
        fit_prompts_per_label=np.array([PROMPTS_PER_LABEL]),
        kfold_splits=np.array([KFOLD_SPLITS]),
        median_layer_cosine=np.array([report["median_layer_cosine"]]),
        p10_layer_cosine=np.array([report["p10_layer_cosine"]]),
        held_out_unsafe_categories=np.array(HELD_OUT_UNSAFE_CATEGORIES),
    )

    print(f"wrote split: {SPLIT_PATH}", flush=True)
    print(f"wrote fit report: {FIT_REPORT_PATH}", flush=True)
    print(f"wrote directions: {DIRECTIONS_PATH}", flush=True)


def choose_fit_rows(rows):
    fit_rows = [
        category_balanced_sample(rows, "safe", PROMPTS_PER_LABEL),
        category_balanced_sample(rows, "unsafe", PROMPTS_PER_LABEL),
    ]
    return pd.concat(fit_rows).sample(frac=1, random_state=RANDOM_SEED)


def category_balanced_sample(rows, label, total):
    rows = rows.loc[rows["label"] == label]
    categories = sorted(rows["category"].unique())
    base = total // len(categories)
    extra = total % len(categories)

    # Spread the fixed sample across categories instead of letting one domain dominate.
    samples = []
    for index, category in enumerate(categories):
        category_rows = rows.loc[rows["category"] == category]
        take = base + int(index < extra)
        samples.append(category_rows.sample(n=take, random_state=RANDOM_SEED + index))

    return pd.concat(samples)


def load_or_make_activations(rows):
    prompts = rows["prompt"].tolist()
    labels = rows["label"].tolist()

    # Activation collection is the slow part, so reuse it only for the exact same split.
    if CONSTRUCTION_ACTIVATIONS_PATH.exists():
        cached = np.load(CONSTRUCTION_ACTIVATIONS_PATH, allow_pickle=False)
        same_prompts = cached["prompts"].astype(str).tolist() == prompts
        same_labels = cached["labels"].astype(str).tolist() == labels
        if same_prompts and same_labels:
            print(f"using cached activations: {CONSTRUCTION_ACTIVATIONS_PATH}", flush=True)
            return cached["activations"]

        raise ValueError("cached activations do not match the current split")

    tokenizer, model, device = load_gemma()
    activations = []
    for index, row in enumerate(rows.itertuples(index=False), start=1):
        print(f"activations {index}/{len(rows)} id={row.id}", flush=True)
        activations.append(last_prompt_token_activations(model, tokenizer, row.prompt, device))

    activations = np.stack(activations, axis=0).astype(np.float32)
    np.savez_compressed(
        CONSTRUCTION_ACTIVATIONS_PATH,
        activations=activations,
        prompts=np.array(prompts),
        labels=np.array(labels),
    )
    return activations


def kfold_report(activations, labels):
    fold_directions = []
    splitter = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # Refit the direction while leaving out each fold once.
    for train_positions, _ in splitter.split(activations, labels):
        fold_directions.append(refusal_directions(activations[train_positions], labels[train_positions]))

    mean_direction = np.mean(np.stack(fold_directions), axis=0)
    mean_direction = mean_direction / np.linalg.norm(mean_direction, axis=1, keepdims=True)

    # Cosine close to 1 means the learned layer directions barely change across folds.
    layer_cosines = np.concatenate([
        np.sum(directions * mean_direction, axis=1)
        for directions in fold_directions
    ])

    median_cosine = float(np.median(layer_cosines))
    p10_cosine = float(np.quantile(layer_cosines, 0.10))

    return {
        "fit_prompts_total": len(labels),
        "fit_prompts_per_label": PROMPTS_PER_LABEL,
        "kfold_splits": KFOLD_SPLITS,
        "median_layer_cosine": median_cosine,
        "p10_layer_cosine": p10_cosine,
        "held_out_unsafe_categories": " | ".join(HELD_OUT_UNSAFE_CATEGORIES),
    }


if __name__ == "__main__":
    main()
