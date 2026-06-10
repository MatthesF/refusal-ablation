from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_validation import (
    apply_weight_edit,
    choose_device,
    format_chat_prompt,
    load_mean_difference_directions,
    load_split_dataset,
    output_quality_flags,
    refusal_label,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate side-by-side baseline vs edited full outputs for a small "
            "paired prompt sample."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/Users/matthesfogtmann/Desktop/statprojekt/project/experiments/teest/data/"
            "subcategory_balanced_prompts.csv"
        ),
    )
    parser.add_argument(
        "--activations",
        type=Path,
        default=Path(
            "/Users/matthesfogtmann/Desktop/statprojekt/project/experiments/arditi_minimal/"
            "artifacts/prompt_last_construction_hidden_states.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/Users/matthesfogtmann/Desktop/statprojekt/project/experiments/"
            "norm_preserving_minimal/artifacts/paired_full_outputs_8unsafe_2safe"
        ),
    )
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--safe-count", type=int, default=2)
    parser.add_argument("--unsafe-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2445)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--direction-mode",
        choices=["mean_diff", "biprojected_mean_diff"],
        default="biprojected_mean_diff",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def select_prompt_sample(df: pd.DataFrame, safe_count: int, unsafe_count: int, seed: int) -> pd.DataFrame:
    pieces = []
    for category, count in [("safe", safe_count), ("unsafe", unsafe_count)]:
        candidates = (
            df[df["category"] == category]
            .sort_values(["subcategory", "id"])
            .groupby("subcategory", as_index=False)
            .head(1)
        )
        if len(candidates) < count:
            raise ValueError(
                f"Need {count} {category} subcategory representatives, found {len(candidates)}"
            )
        pieces.append(candidates.sample(n=count, random_state=seed + len(category)))
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_outputs(
    model: torch.nn.Module,
    tokenizer,
    prompts: pd.DataFrame,
    condition: str,
    model_id: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> list[dict]:
    rows = []
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        for prompt in prompts.itertuples(index=False):
            chat_text = format_chat_prompt(tokenizer, prompt.prompt)
            encoded = tokenizer(chat_text, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(**encoded, **generation_kwargs)
            prompt_length = encoded["input_ids"].shape[1]
            new_token_ids = generated[0, prompt_length:]
            output_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
            stop_reason = "max_new_tokens" if len(new_token_ids) >= max_new_tokens else "eos_or_stop"
            quality = output_quality_flags(output_text, len(new_token_ids), max_new_tokens)
            rows.append(
                {
                    "id": prompt.id,
                    "split": prompt.split,
                    "category": prompt.category,
                    "subcategory": prompt.subcategory,
                    "condition": condition,
                    "model_id": model_id,
                    "prompt": prompt.prompt,
                    "output_text": output_text,
                    "heuristic_refused": refusal_label(output_text),
                    "stop_reason": stop_reason,
                    **quality,
                }
            )
            print(f"generated {condition} id={prompt.id}", flush=True)
    return rows


def make_side_by_side(rows: pd.DataFrame) -> pd.DataFrame:
    paired_rows = []
    for prompt_id, pair in rows.groupby("id", sort=False):
        by_condition = pair.set_index("condition")
        missing = {"baseline", "edited"} - set(by_condition.index)
        if missing:
            raise ValueError(f"Missing paired outputs for id={prompt_id}: {sorted(missing)}")
        baseline = by_condition.loc["baseline"]
        edited = by_condition.loc["edited"]
        paired_rows.append(
            {
                "id": prompt_id,
                "split": baseline["split"],
                "category": baseline["category"],
                "subcategory": baseline["subcategory"],
                "prompt": baseline["prompt"],
                "baseline_output_text": baseline["output_text"],
                "edited_output_text": edited["output_text"],
                "baseline_heuristic_refused": baseline["heuristic_refused"],
                "edited_heuristic_refused": edited["heuristic_refused"],
                "baseline_new_tokens": baseline["new_tokens"],
                "edited_new_tokens": edited["new_tokens"],
                "baseline_stop_reason": baseline["stop_reason"],
                "edited_stop_reason": edited["stop_reason"],
                "baseline_output_chars": baseline["output_chars"],
                "edited_output_chars": edited["output_chars"],
                "baseline_repetition_flag": baseline["repetition_flag"],
                "edited_repetition_flag": edited["repetition_flag"],
            }
        )
    return pd.DataFrame(paired_rows)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_df = load_split_dataset(args.dataset, args.split)
    prompts = select_prompt_sample(split_df, args.safe_count, args.unsafe_count, args.seed)
    prompts.to_csv(args.output_dir / "selected_prompts.csv", index=False)

    directions, _layer_indices = load_mean_difference_directions(args.activations, args.direction_mode)
    device = choose_device() if args.device == "auto" else args.device
    dtype = torch.float32 if device == "cpu" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    baseline_rows = generate_outputs(
        model,
        tokenizer,
        prompts,
        "baseline",
        args.model_id,
        device,
        args.max_new_tokens,
        args.temperature,
    )

    edit_stats = apply_weight_edit(model, directions, args.alpha)
    edited_rows = generate_outputs(
        model,
        tokenizer,
        prompts,
        "edited",
        args.model_id,
        device,
        args.max_new_tokens,
        args.temperature,
    )

    long_df = pd.DataFrame(baseline_rows + edited_rows)
    side_by_side = make_side_by_side(long_df)

    long_path = args.output_dir / "paired_full_outputs_long.csv"
    side_by_side_path = args.output_dir / "paired_full_outputs_side_by_side.csv"
    edit_stats_path = args.output_dir / "edit_stats.csv"

    long_df.to_csv(long_path, index=False)
    side_by_side.to_csv(side_by_side_path, index=False)
    pd.DataFrame(edit_stats).to_csv(edit_stats_path, index=False)

    summary = {
        "n_prompts": int(len(side_by_side)),
        "safe_prompts": int((side_by_side["category"] == "safe").sum()),
        "unsafe_prompts": int((side_by_side["category"] == "unsafe").sum()),
        "baseline_refused": int(side_by_side["baseline_heuristic_refused"].sum()),
        "edited_refused": int(side_by_side["edited_heuristic_refused"].sum()),
        "baseline_cap_hits": int((side_by_side["baseline_stop_reason"] == "max_new_tokens").sum()),
        "edited_cap_hits": int((side_by_side["edited_stop_reason"] == "max_new_tokens").sum()),
        "baseline_repetition": int(side_by_side["baseline_repetition_flag"].sum()),
        "edited_repetition": int(side_by_side["edited_repetition_flag"].sum()),
        "side_by_side_csv": str(side_by_side_path),
        "long_csv": str(long_path),
    }
    pd.Series(summary).to_json(args.output_dir / "summary.json", indent=2)
    print(summary)


if __name__ == "__main__":
    main()
