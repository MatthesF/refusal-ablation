import numpy as np
import pandas as pd

from settings import (
    ACTIVATION_BATCH_SIZE,
    CATEGORY_GENERALIZATION_P10_QUANTILE,
    CATEGORY_GENERALIZATION_PATH,
    CATEGORY_GENERALIZATION_POLICY_AREAS,
    CATEGORY_GENERALIZATION_REFERENCE,
    CATEGORY_GENERALIZATION_REPEATS,
    CONSTRUCTION_POLICY_AREAS,
    DIRECTION_SPLIT_STYLE,
    DIRECTIONS_PATH,
    FIT_REPORT_PATH,
    GEMMA_MODEL_ID,
    GEMMA_MODEL_REVISION,
    POLICY_ACTIVATIONS_PATH,
    RANDOM_SEED,
    RUN_DIR,
    SPLIT_PATH,
)
from src.dataset import load_prompts, split_prompts
from src.gemma import last_prompt_token_activation_batch, load_gemma, refusal_directions


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # This split is the run contract: fitting, generation, and reporting all
    # refer back to these prompt ids.
    split_rows = split_prompts(load_prompts())
    split_rows.to_csv(SPLIT_PATH, index=False)

    labels = split_rows["label"].to_numpy()
    categories = split_rows["category"].to_numpy()
    activations = load_or_make_activations(split_rows)

    # The generalization table is diagnostic only. It asks whether directions
    # fit on some policy areas point the same way as directions from held-out
    # policy areas; the final edit below still uses the fixed construction split.
    generalization = category_generalization_table(activations, labels, categories)
    generalization.to_csv(CATEGORY_GENERALIZATION_PATH, index=False)

    construction_mask = split_rows["split"].eq("construction").to_numpy()
    construction_rows = split_rows.loc[construction_mask].reset_index(drop=True)
    construction_activations = activations[construction_mask]
    construction_labels = labels[construction_mask]

    final_budget_rows = generalization.loc[
        generalization["used_policy_areas"] == CONSTRUCTION_POLICY_AREAS
    ]
    if final_budget_rows.empty:
        raise ValueError(
            "category generalization table must include the final construction policy-area count"
        )
    final_budget_row = final_budget_rows.iloc[0]
    report = {
        "split_style": DIRECTION_SPLIT_STYLE,
        "construction_prompts": len(construction_labels),
        "construction_safe_prompts": int(np.sum(construction_labels == "safe")),
        "construction_unsafe_prompts": int(np.sum(construction_labels == "unsafe")),
        "construction_policy_areas": CONSTRUCTION_POLICY_AREAS,
        "category_generalization_repeats": CATEGORY_GENERALIZATION_REPEATS,
        "category_generalization_reference": CATEGORY_GENERALIZATION_REFERENCE,
        "median_layer_cosine_to_heldout": float(final_budget_row["median_layer_cosine_to_heldout"]),
        "p10_layer_cosine_to_heldout": float(final_budget_row["p10_layer_cosine_to_heldout"]),
    }
    pd.DataFrame([report]).to_csv(FIT_REPORT_PATH, index=False)

    # One final direction per layer, fit only from the no-leakage construction areas.
    directions = refusal_directions(construction_activations, construction_labels)
    np.savez_compressed(
        DIRECTIONS_PATH,
        directions=directions,
        split_style=np.array([DIRECTION_SPLIT_STYLE]),
        model_id=np.array([GEMMA_MODEL_ID]),
        model_revision=np.array([GEMMA_MODEL_REVISION]),
        construction_categories=np.array(sorted(construction_rows["category"].unique())),
        construction_prompt_ids=construction_rows["id"].to_numpy(),
        construction_policy_areas=np.array([CONSTRUCTION_POLICY_AREAS]),
        category_generalization_repeats=np.array([CATEGORY_GENERALIZATION_REPEATS]),
        category_generalization_reference=np.array([CATEGORY_GENERALIZATION_REFERENCE]),
        median_layer_cosine_to_heldout=np.array([report["median_layer_cosine_to_heldout"]]),
        p10_layer_cosine_to_heldout=np.array([report["p10_layer_cosine_to_heldout"]]),
    )

    print(f"wrote split: {SPLIT_PATH}", flush=True)
    print(f"wrote category generalization data: {CATEGORY_GENERALIZATION_PATH}", flush=True)
    print(f"wrote fit report: {FIT_REPORT_PATH}", flush=True)
    print(f"wrote directions: {DIRECTIONS_PATH}", flush=True)


def load_or_make_activations(rows):
    prompt_ids = rows["id"].to_numpy()
    prompts = rows["prompt"].tolist()
    labels = rows["label"].tolist()
    categories = rows["category"].tolist()

    if POLICY_ACTIVATIONS_PATH.exists():
        with np.load(POLICY_ACTIVATIONS_PATH, allow_pickle=False) as cached:
            # Cached activations are valid only for the exact same prompt rows.
            expected_cache_files = {"activations", "prompt_ids", "prompts", "labels", "categories"}
            cache_has_metadata = expected_cache_files.issubset(cached.files)
            cache_matches_rows = (
                cache_has_metadata
                and cached["prompt_ids"].tolist() == prompt_ids.tolist()
                and cached["prompts"].astype(str).tolist() == prompts
                and cached["labels"].astype(str).tolist() == labels
                and cached["categories"].astype(str).tolist() == categories
            )
            if cache_matches_rows:
                print(f"using cached activations: {POLICY_ACTIVATIONS_PATH}", flush=True)
                return cached["activations"]
        print("cached activations do not match the current prompts; rebuilding", flush=True)

    tokenizer, model, device = load_gemma()
    activations = []
    for start in range(0, len(rows), ACTIVATION_BATCH_SIZE):
        batch = rows.iloc[start:start + ACTIVATION_BATCH_SIZE]
        end = start + len(batch)
        first_id = batch["id"].iloc[0]
        last_id = batch["id"].iloc[-1]
        print(f"activations {start + 1}-{end}/{len(rows)} ids={first_id}-{last_id}", flush=True)
        activations.append(
            last_prompt_token_activation_batch(
                model,
                tokenizer,
                batch["prompt"].tolist(),
                device,
            )
        )

    activations = np.concatenate(activations, axis=0).astype(np.float32)
    np.savez_compressed(
        POLICY_ACTIVATIONS_PATH,
        activations=activations,
        prompt_ids=prompt_ids,
        prompts=np.array(prompts),
        labels=np.array(labels),
        categories=np.array(categories),
    )
    return activations


def category_generalization_table(activations, labels, categories):
    policy_areas = np.array(sorted(set(categories)))
    policy_area_count = len(policy_areas)
    bad_budgets = [
        used_policy_areas
        for used_policy_areas in CATEGORY_GENERALIZATION_POLICY_AREAS
        if used_policy_areas <= 0 or used_policy_areas >= policy_area_count
    ]
    if bad_budgets:
        raise ValueError(
            "category generalization budgets must be between "
            f"1 and {policy_area_count - 1}: {bad_budgets}"
        )

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for used_policy_areas in CATEGORY_GENERALIZATION_POLICY_AREAS:
        split_cosines = []
        used_prompt_counts = []
        held_out_prompt_counts = []

        for _ in range(CATEGORY_GENERALIZATION_REPEATS):
            used_categories = rng.choice(policy_areas, size=used_policy_areas, replace=False)
            held_out_categories = np.setdiff1d(policy_areas, used_categories)
            used_mask = np.isin(categories, used_categories)
            held_out_mask = np.isin(categories, held_out_categories)
            used_direction = refusal_directions(activations[used_mask], labels[used_mask])
            held_out_direction = refusal_directions(
                activations[held_out_mask],
                labels[held_out_mask],
            )

            split_cosines.append(np.sum(used_direction * held_out_direction, axis=1))
            used_prompt_counts.append(int(np.sum(used_mask)))
            held_out_prompt_counts.append(int(np.sum(held_out_mask)))

        split_cosines = np.concatenate(split_cosines)

        rows.append({
            "used_policy_areas": int(used_policy_areas),
            "held_out_policy_areas": len(policy_areas) - used_policy_areas,
            "used_prompts": int(np.mean(used_prompt_counts)),
            "held_out_prompts": int(np.mean(held_out_prompt_counts)),
            "repeats": CATEGORY_GENERALIZATION_REPEATS,
            "cosine_reference": CATEGORY_GENERALIZATION_REFERENCE,
            "median_layer_cosine_to_heldout": float(np.median(split_cosines)),
            "p10_layer_cosine_to_heldout": float(
                np.quantile(split_cosines, CATEGORY_GENERALIZATION_P10_QUANTILE)
            ),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
