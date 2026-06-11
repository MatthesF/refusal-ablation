import pandas as pd
from sklearn.model_selection import train_test_split

from settings import (
    CONSTRUCTION_POLICY_AREAS,
    DATASET_PATH,
    POLICY_AREAS,
    PROMPTS_PER_LABEL_PER_POLICY_AREA,
    RANDOM_SEED,
)


def load_prompts():
    prompts = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    required_columns = {"prompt", "label", "category"}
    missing_columns = required_columns - set(prompts.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"dataset is missing required columns: {missing}")

    columns = ["prompt", "label", "category"]
    if "policy_area" in prompts.columns:
        columns.append("policy_area")
    prompts = prompts[columns].copy()

    # Keep one stable id per prompt so later output CSVs can be merged exactly.
    prompts.insert(0, "id", range(len(prompts)))

    for column in columns:
        prompts[column] = prompts[column].astype(str).str.strip()

    labels = set(prompts["label"])
    if labels != {"safe", "unsafe"}:
        raise ValueError("dataset labels must be exactly: safe, unsafe")

    validate_policy_dataset(prompts)

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
    categories = prompts[["category"]].drop_duplicates().sort_values("category")

    # Split policy areas, not prompts. This is the no-leakage part of the design.
    construction, _ = train_test_split(
        categories,
        train_size=CONSTRUCTION_POLICY_AREAS,
        random_state=RANDOM_SEED,
    )
    return set(construction["category"])


def validate_policy_dataset(prompts):
    categories = set(prompts["category"])
    expected_categories = set(POLICY_AREAS)
    if categories != expected_categories:
        missing = sorted(expected_categories - categories)
        extra = sorted(categories - expected_categories)
        raise ValueError(f"dataset policy areas mismatch; missing={missing}, extra={extra}")

    counts = (
        prompts.groupby(["category", "label"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    expected_count = PROMPTS_PER_LABEL_PER_POLICY_AREA
    bad = counts.loc[
        (counts["safe"] != expected_count)
        | (counts["unsafe"] != expected_count)
    ]
    if not bad.empty:
        raise ValueError(
            "each policy area must contain exactly "
            f"{expected_count} safe and {expected_count} unsafe prompts"
        )

    if "policy_area" in prompts.columns and prompts["policy_area"].str.len().eq(0).any():
        raise ValueError("policy_area must be non-empty for every row")
