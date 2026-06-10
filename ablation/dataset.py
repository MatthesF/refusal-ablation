import random

import pandas as pd

from settings import (
    CONSTRUCTION_PER_CATEGORY,
    DATASET_PATH,
    HELD_OUT_UNSAFE_CATEGORIES,
    RANDOM_SEED,
)


def load_prompts():
    prompts = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    required = {"prompt", "label", "category"}
    if required - set(prompts.columns):
        raise ValueError("dataset must have columns: prompt,label,category")

    # Keep a stable id so generated outputs can be matched back to the same prompt.
    prompts = prompts[["prompt", "label", "category"]].copy()
    prompts.insert(0, "id", range(len(prompts)))

    for column in ["prompt", "label", "category"]:
        prompts[column] = prompts[column].astype(str).str.strip()

    labels = set(prompts["label"])
    if labels != {"safe", "unsafe"}:
        raise ValueError("dataset labels must be exactly: safe, unsafe")

    unsafe_categories = set(prompts.loc[prompts["label"] == "unsafe", "category"])
    missing = set(HELD_OUT_UNSAFE_CATEGORIES) - unsafe_categories
    if missing:
        raise ValueError(f"held-out unsafe categories missing from dataset: {sorted(missing)}")

    return prompts


def split_prompts(prompts):
    rng = random.Random(RANDOM_SEED)
    pieces = []

    # Split inside each category so construction and test keep the same domain balance.
    for category, rows in prompts.groupby("category", sort=True):
        labels = rows["label"].unique()
        if len(labels) != 1:
            raise ValueError(f"category has mixed labels: {category}")

        # Use Python's seeded shuffle so the split stays identical to the activation cache.
        shuffled_index = list(rows.index)
        rng.shuffle(shuffled_index)
        rows = rows.loc[shuffled_index]
        label = labels[0]

        # These domains are held out completely to test cross-category generalization.
        if label == "unsafe" and category in HELD_OUT_UNSAFE_CATEGORIES:
            pieces.append(rows.assign(split="test", test_group="heldout_unsafe"))
            continue

        if len(rows) <= CONSTRUCTION_PER_CATEGORY:
            raise ValueError(f"not enough prompts in category: {category}")

        # Non-held-out categories give a small construction slice; the rest is test data.
        test_group = "safe" if label == "safe" else "seen_unsafe"
        pieces.append(rows.head(CONSTRUCTION_PER_CATEGORY).assign(
            split="construction",
            test_group="construction",
        ))
        pieces.append(rows.iloc[CONSTRUCTION_PER_CATEGORY:].assign(
            split="test",
            test_group=test_group,
        ))

    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values("id")
        .reset_index(drop=True)
    )
