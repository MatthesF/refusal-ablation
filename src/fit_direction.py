from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from settings import (
    CONSTRUCTION_ACTIVATIONS_PATH,
    CONSTRUCTION_CATEGORIES_PER_LABEL,
    DIRECTIONS_PATH,
    MODEL_ID,
    MODEL_REVISION,
    RANDOM_SEED,
    RUN_DIR,
    SPLIT_PATH,
    STABILITY_FOLDS,
)
from src.dataset import load_prompts, split_prompts
from src.model import last_prompt_token_activations, load_gemma, refusal_directions


FIT_REPORT_PATH = RUN_DIR / "direction_fit_report.csv"
SPLIT_STYLE = "category_disjoint"


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # The split file is the contract between fitting, generation, and manual labeling.
    split_rows = split_prompts(load_prompts())
    split_rows.to_csv(SPLIT_PATH, index=False)

    construction_rows = split_rows.loc[split_rows["split"] == "construction"].reset_index(drop=True)
    labels = construction_rows["label"].to_numpy()
    activations = load_or_make_activations(construction_rows)

    # The report uses this as a stability check, not as model selection.
    report = stability_report(activations, labels, construction_rows["category"].to_numpy())
    pd.DataFrame([report]).to_csv(FIT_REPORT_PATH, index=False)

    # The final edit is fitted once, using every construction prompt.
    directions = refusal_directions(activations, labels)
    np.savez_compressed(
        DIRECTIONS_PATH,
        directions=directions,
        split_style=np.array([SPLIT_STYLE]),
        model_id=np.array([MODEL_ID]),
        model_revision=np.array([MODEL_REVISION]),
        construction_categories=np.array(sorted(construction_rows["category"].unique())),
        construction_prompt_ids=construction_rows["id"].to_numpy(),
        construction_categories_per_label=np.array([CONSTRUCTION_CATEGORIES_PER_LABEL]),
        stability_folds=np.array([STABILITY_FOLDS]),
        median_pairwise_layer_cosine=np.array([report["median_pairwise_layer_cosine"]]),
        p10_pairwise_layer_cosine=np.array([report["p10_pairwise_layer_cosine"]]),
    )

    print(f"wrote split: {SPLIT_PATH}", flush=True)
    print(f"wrote fit report: {FIT_REPORT_PATH}", flush=True)
    print(f"wrote directions: {DIRECTIONS_PATH}", flush=True)


def load_or_make_activations(rows):
    prompts = rows["prompt"].tolist()
    labels = rows["label"].tolist()
    categories = rows["category"].tolist()

    if CONSTRUCTION_ACTIVATIONS_PATH.exists():
        with np.load(CONSTRUCTION_ACTIVATIONS_PATH, allow_pickle=False) as cached:
            # Cached activations are valid only for the exact same construction rows.
            if cache_matches(cached, prompts, labels, categories):
                print(f"using cached activations: {CONSTRUCTION_ACTIVATIONS_PATH}", flush=True)
                return cached["activations"]
        print("cached activations are stale; rebuilding them for the current split", flush=True)

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
        categories=np.array(categories),
    )
    return activations


def cache_matches(cached, prompts, labels, categories):
    expected = {"activations", "prompts", "labels", "categories"}
    if not expected.issubset(cached.files):
        return False

    return (
        cached["prompts"].astype(str).tolist() == prompts
        and cached["labels"].astype(str).tolist() == labels
        and cached["categories"].astype(str).tolist() == categories
    )


def stability_report(activations, labels, categories):
    splitter = StratifiedKFold(n_splits=STABILITY_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_directions = []
    fold_sizes = []

    # Each fold has prompts from every construction category, but no prompt overlap.
    for _, fold_positions in splitter.split(activations, categories):
        fold_directions.append(refusal_directions(activations[fold_positions], labels[fold_positions]))
        fold_sizes.append(len(fold_positions))

    # Five fold directions give ten pairwise comparisons per layer.
    pairwise_layer_cosines = []
    for first, second in combinations(fold_directions, 2):
        pairwise_layer_cosines.append(np.sum(first * second, axis=1))
    pairwise_layer_cosines = np.concatenate(pairwise_layer_cosines)

    return {
        "split_style": SPLIT_STYLE,
        "construction_prompts": len(labels),
        "construction_safe_prompts": int(np.sum(labels == "safe")),
        "construction_unsafe_prompts": int(np.sum(labels == "unsafe")),
        "construction_categories_per_label": CONSTRUCTION_CATEGORIES_PER_LABEL,
        "stability_folds": STABILITY_FOLDS,
        "stability_fold_prompts": " | ".join(str(size) for size in fold_sizes),
        "pairwise_comparisons": len(fold_directions) * (len(fold_directions) - 1) // 2,
        "median_pairwise_layer_cosine": float(np.median(pairwise_layer_cosines)),
        "p10_pairwise_layer_cosine": float(np.quantile(pairwise_layer_cosines, 0.10)),
    }


if __name__ == "__main__":
    main()
