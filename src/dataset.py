import pandas as pd
from sklearn.model_selection import train_test_split

from settings import CONSTRUCTION_CATEGORIES_PER_LABEL, DATASET_PATH, RANDOM_SEED


def load_prompts():
    prompts = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    required = {"prompt", "label", "category"}
    missing = required - set(prompts.columns)
    if missing:
        raise ValueError("dataset must have columns: prompt,label,category")

    prompts = prompts[["prompt", "label", "category"]].copy()

    # Keep one stable id per prompt so later output CSVs can be merged exactly.
    prompts.insert(0, "id", range(len(prompts)))

    for column in ["prompt", "label", "category"]:
        prompts[column] = prompts[column].astype(str).str.strip()

    labels = set(prompts["label"])
    if labels != {"safe", "unsafe"}:
        raise ValueError("dataset labels must be exactly: safe, unsafe")

    mixed = prompts.groupby("category")["label"].nunique()
    mixed = mixed[mixed > 1].index.tolist()
    if mixed:
        raise ValueError(f"categories must not mix labels: {mixed}")

    return prompts


def split_prompts(prompts):
    construction_categories = choose_construction_categories(prompts)

    split = prompts.copy()
    split["split"] = "test"
    split["test_group"] = split["label"]

    # A prompt is either used to build the edit or to evaluate it, never both.
    construction = split["category"].isin(construction_categories)
    split.loc[construction, "split"] = "construction"
    split.loc[construction, "test_group"] = "construction"

    return split.sort_values("id").reset_index(drop=True)


def choose_construction_categories(prompts):
    # The split happens at domain level: one row per category.
    categories = (
        prompts[["category", "label"]]
        .drop_duplicates()
        .sort_values(["label", "category"])
        .reset_index(drop=True)
    )
    construction_count = 2 * CONSTRUCTION_CATEGORIES_PER_LABEL

    # Split categories, not prompts. This is the no-leakage part of the design.
    construction, _ = train_test_split(
        categories,
        train_size=construction_count,
        stratify=categories["label"],
        random_state=RANDOM_SEED,
    )

    counts = construction["label"].value_counts().to_dict()
    expected = {"safe": CONSTRUCTION_CATEGORIES_PER_LABEL, "unsafe": CONSTRUCTION_CATEGORIES_PER_LABEL}
    if counts != expected:
        raise ValueError(f"construction category split is not balanced: {counts}")

    return set(construction["category"])
