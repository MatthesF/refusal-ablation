import hashlib
import importlib.util
import json
import subprocess
import sys
import urllib.request

import pandas as pd

from settings import (
    DIRECTIONS_PATH,
    GENERATION_MAX_NEW_TOKENS,
    ROOT,
    SORRY_BENCH_DIR,
    SORRY_BENCH_JUDGE_LOCAL_DIR,
    SORRY_BENCH_JUDGE_MODEL_NAME,
    SORRY_BENCH_OFFICIAL_COMMIT,
    SORRY_BENCH_OFFICIAL_DIR,
    SORRY_BENCH_OFFICIAL_JUDGE_PROMPT_NAME,
    SORRY_BENCH_OFFICIAL_JUDGMENTS_PATH,
    SORRY_BENCH_OFFICIAL_SUMMARY_PATH,
    SORRY_BENCH_OUTPUTS_PATH,
    SORRY_BENCH_QUESTION_PATH,
)


BASELINE_MODEL_ID = "gemma_baseline"
EDITED_MODEL_ID = "gemma_edited"
OFFICIAL_JUDGE_FILENAME = f"{SORRY_BENCH_JUDGE_MODEL_NAME}.jsonl"

# These hashes are the guardrail around vendored evaluator code: if upstream
# changes or a download is corrupted, the run stops before scoring.
OFFICIAL_FILE_HASHES = {
    "LICENSE": "1a9a3130473de25fcb91872a735c8844e830a26a916e2177e1357e4958af0300",
    "README.md": "c97051098178e08274df4fe2108cec09f1a7f06d73e8ac4490cbe751e69d4ca7",
    "common.py": "7f29b19a36237b9e0ced4690f83cc9b69acc78d9bf13fe023ed99a097a141c8b",
    "gen_judgment_safety.py": "6384c5249e777120685daa06fb6a25fc52c207c20d82837bb4eaef76f45fe68f",
    "gen_judgment_safety_vllm.py": "3605157a82c81fa53abfa461addbf13a3e83de2797c7b4a1637153ce38893d39",
    "gen_model_answer.py": "5dfb8a0ba929757129c73b98bdc5c248b57d6ad1ebc1be4cdc54f3ec101b6e76",
    "gen_model_answer_vllm.py": "cf343d154e1aec94933faf6180befb4067cfa1574fcec144a04d6c531044d817",
    "data/sorry_bench/judge_prompts.jsonl": "c0760e55a1f8ef1124c6bc3ce80392fbcb4be626f9c4f8451c5e030c98f5ca9c",
}


def current_run_id():
    if not DIRECTIONS_PATH.exists():
        raise ValueError("run python -m src.fit_direction before SORRY-Bench generation")
    direction_hash = hashlib.sha256(DIRECTIONS_PATH.read_bytes()).hexdigest()[:12]
    return f"{direction_hash}-sorry-bench-base-max{GENERATION_MAX_NEW_TOKENS}"


def require_questions():
    if not SORRY_BENCH_QUESTION_PATH.exists():
        raise FileNotFoundError(
            "SORRY-Bench question file is missing. Request access to "
            "https://huggingface.co/datasets/sorry-bench/sorry-bench-202503, "
            "export HF_TOKEN if needed, then run: python -m src.download_assets"
        )


def load_questions():
    path = SORRY_BENCH_QUESTION_PATH
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as question_file:
        for line_number, line in enumerate(question_file, start=1):
            if not line.strip():
                continue
            row = normalize_question(json.loads(line), line_number)
            if row["question_id"] in seen:
                raise ValueError(f"duplicate SORRY-Bench question_id: {row['question_id']}")
            seen.add(row["question_id"])
            rows.append(row)

    if not rows:
        raise ValueError(f"SORRY-Bench question file has no rows: {path}")
    return pd.DataFrame(rows).sort_values("question_id").reset_index(drop=True)


def normalize_question(record, line_number):
    required = {"question_id", "category", "turns"}
    missing = required - set(record)
    if missing:
        raise ValueError(
            f"SORRY-Bench row {line_number} is missing fields: {', '.join(sorted(missing))}"
        )

    turns = record["turns"]
    if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], str):
        raise ValueError(f"SORRY-Bench row {line_number} must contain exactly one text turn")

    prompt = turns[0].strip()
    if not prompt:
        raise ValueError(f"SORRY-Bench row {line_number} has an empty prompt")

    try:
        question_id = int(record["question_id"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"SORRY-Bench row {line_number} has a non-integer question_id") from error

    return {
        "question_id": question_id,
        "category": str(record["category"]).strip(),
        "prompt": prompt,
    }


def download_official_evaluator():
    for relative_path, expected_hash in OFFICIAL_FILE_HASHES.items():
        url = (
            "https://raw.githubusercontent.com/SORRY-Bench/sorry-bench/"
            f"{SORRY_BENCH_OFFICIAL_COMMIT}/{relative_path}"
        )
        with urllib.request.urlopen(url) as response:
            data = response.read()

        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"official SORRY-Bench hash mismatch for {relative_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        destination = SORRY_BENCH_OFFICIAL_DIR / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    (SORRY_BENCH_OFFICIAL_DIR / "UPSTREAM_COMMIT").write_text(
        SORRY_BENCH_OFFICIAL_COMMIT + "\n",
        encoding="utf-8",
    )


def require_official_evaluator():
    missing = []
    mismatched = []
    for relative_path, expected_hash in OFFICIAL_FILE_HASHES.items():
        path = SORRY_BENCH_OFFICIAL_DIR / relative_path
        if not path.exists():
            missing.append(relative_path)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            mismatched.append(relative_path)

    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if mismatched:
            details.append(f"hash mismatch: {', '.join(mismatched)}")
        raise FileNotFoundError(
            "official SORRY-Bench evaluator is not ready; run python -m src.download_assets. "
            + "; ".join(details)
        )


def official_judge_prompt():
    require_official_evaluator()
    prompt_path = SORRY_BENCH_OFFICIAL_DIR / "data/sorry_bench/judge_prompts.jsonl"
    with prompt_path.open(encoding="utf-8") as prompt_file:
        for line in prompt_file:
            prompt = json.loads(line)
            if prompt["name"] == SORRY_BENCH_OFFICIAL_JUDGE_PROMPT_NAME:
                return prompt
    raise ValueError(f"official judge prompt not found: {SORRY_BENCH_OFFICIAL_JUDGE_PROMPT_NAME}")


def export_official_answers():
    outputs = pd.read_csv(SORRY_BENCH_OUTPUTS_PATH)
    run_id = current_run_id()
    rows = outputs.loc[outputs["run_id"] == run_id].copy()
    if rows.empty:
        raise ValueError(f"no SORRY-Bench outputs found for current run_id: {run_id}")

    expected_question_ids = set(load_questions()["question_id"])
    manifest_rows = []
    for condition, model_id in [("baseline", BASELINE_MODEL_ID), ("edited", EDITED_MODEL_ID)]:
        # The official judge reads one JSONL answer file per model id.
        condition_rows = rows.loc[rows["condition"] == condition].sort_values("question_id")
        if condition_rows.empty:
            raise ValueError(f"missing SORRY-Bench outputs for condition: {condition}")

        duplicated = condition_rows.loc[
            condition_rows["question_id"].duplicated(),
            "question_id",
        ].tolist()
        if duplicated:
            raise ValueError(f"duplicate SORRY-Bench outputs for {condition}: {duplicated[:10]}")

        actual_question_ids = set(condition_rows["question_id"])
        missing = sorted(expected_question_ids - actual_question_ids)
        extra = sorted(actual_question_ids - expected_question_ids)
        if missing or extra:
            raise ValueError(
                f"incomplete SORRY-Bench outputs for {condition}: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

        answer_path = SORRY_BENCH_DIR / "model_answer" / f"{model_id}.jsonl"
        answer_path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            official_answer_record(row, model_id)
            for row in condition_rows.itertuples(index=False)
        ]
        answer_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        manifest_rows.append({
            "run_id": run_id,
            "condition": condition,
            "model_id": model_id,
            "answer_file": str(answer_path),
            "rows": len(records),
        })

    return pd.DataFrame(manifest_rows)


def official_answer_record(row, model_id):
    answer = "" if pd.isna(row.answer) else str(row.answer)
    # Deterministic answer ids make regenerated exports diff cleanly.
    answer_id_material = f"{row.run_id}|{row.condition}|{row.question_id}|{answer}"
    return {
        "question_id": int(row.question_id),
        "answer_id": hashlib.sha256(answer_id_material.encode("utf-8")).hexdigest()[:22],
        "model_id": model_id,
        "choices": [{"index": 0, "turns": [answer]}],
        "tstamp": 0,
    }


def run_official_judge(manifest):
    prompt = official_judge_prompt()
    require_official_judge_model()
    require_official_runtime()

    judgment_path = SORRY_BENCH_DIR / "model_judgment" / OFFICIAL_JUDGE_FILENAME
    if judgment_path.exists():
        # The upstream script appends to its judgment file; remove the old one so
        # each score step imports exactly this run's judgments.
        judgment_path.unlink()

    command = [
        sys.executable,
        str(SORRY_BENCH_OFFICIAL_DIR / "gen_judgment_safety_vllm.py"),
        "--bench-name",
        "sorry_bench",
        "--judge-model",
        SORRY_BENCH_JUDGE_MODEL_NAME,
        "--judge-file",
        str(SORRY_BENCH_OFFICIAL_DIR / "data/sorry_bench/judge_prompts.jsonl"),
        "--model-list",
        *manifest["model_id"].tolist(),
    ]
    print(f"using official judge prompt: {prompt['name']}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return command


def require_official_judge_model():
    config_path = SORRY_BENCH_JUDGE_LOCAL_DIR / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "official SORRY-Bench judge checkpoint is missing. "
            "Run python -m src.download_official_judge before scoring. "
            f"Expected: {SORRY_BENCH_JUDGE_LOCAL_DIR}"
        )


def require_official_runtime():
    missing = [
        module
        for module in ["vllm", "fastchat", "openai", "anthropic"]
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "official SORRY-Bench judging requires RunPod evaluator dependencies. "
            f"Missing modules: {', '.join(missing)}. Install with: "
            "python -m pip install -r requirements-official-evaluator.txt"
        )


def import_official_judgments(manifest):
    path = SORRY_BENCH_DIR / "model_judgment" / OFFICIAL_JUDGE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"official SORRY-Bench judgment file is missing: {path}")

    model_to_condition = dict(zip(manifest["model_id"], manifest["condition"]))
    questions = load_questions()
    expected_question_ids = set(questions["question_id"])
    categories = dict(zip(questions["question_id"], questions["category"]))
    rows = []

    with path.open(encoding="utf-8") as judgment_file:
        for line in judgment_file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["model"] not in model_to_condition:
                continue

            raw_score = float(record["score"])
            if raw_score not in {0.0, 1.0}:
                raise ValueError(
                    f"official SORRY-Bench score is outside {{0, 1}}: {record['score']}"
                )
            score = int(raw_score)
            if "tstamp" not in record:
                raise ValueError("official SORRY-Bench judgment is missing tstamp")

            question_id = int(record["question_id"])
            if question_id not in categories:
                raise ValueError(
                    f"official SORRY-Bench judgment has unknown question_id: {question_id}"
                )
            rows.append({
                "run_id": manifest["run_id"].iloc[0],
                "question_id": question_id,
                "condition": model_to_condition[record["model"]],
                "category": categories[question_id],
                "judge_model_id": SORRY_BENCH_JUDGE_MODEL_NAME,
                "judge_prompt_name": SORRY_BENCH_OFFICIAL_JUDGE_PROMPT_NAME,
                "score": score,
                "score_label": "unsafe_compliance" if score == 1 else "refusal",
                "judgment": record["judgment"],
                "official_model_id": record["model"],
                "official_tstamp": record["tstamp"],
            })

    if not rows:
        raise ValueError(
            "no official SORRY-Bench judgments matched Gemma baseline or edited outputs"
        )

    judgments = pd.DataFrame(rows)
    # If a scorer was interrupted and restarted, keep the newest judgment for
    # each answer rather than failing on duplicate official rows.
    judgments = judgments.sort_values("official_tstamp").drop_duplicates(
        ["run_id", "condition", "question_id"],
        keep="last",
    )
    judgments = judgments.drop(columns=["official_tstamp"]).sort_values(
        ["condition", "question_id"]
    )
    for condition in manifest["condition"].tolist():
        condition_rows = judgments.loc[judgments["condition"] == condition]
        actual_question_ids = set(condition_rows["question_id"])
        missing = sorted(expected_question_ids - actual_question_ids)
        extra = sorted(actual_question_ids - expected_question_ids)
        if missing or extra:
            raise ValueError(
                f"incomplete official SORRY-Bench judgments for {condition}: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

    SORRY_BENCH_OFFICIAL_JUDGMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    judgments.to_csv(SORRY_BENCH_OFFICIAL_JUDGMENTS_PATH, index=False)

    summary = summarize_scores(judgments)
    SORRY_BENCH_OFFICIAL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SORRY_BENCH_OFFICIAL_SUMMARY_PATH, index=False)
    return judgments, summary


def summarize_scores(judgments):
    required = {"run_id", "condition", "category", "score"}
    missing = required - set(judgments.columns)
    if missing:
        raise ValueError(f"judgment table is missing columns: {', '.join(sorted(missing))}")

    invalid = judgments.loc[~judgments["score"].isin([0, 1])]
    if not invalid.empty:
        raise ValueError("judgment table contains scores outside the SORRY-Bench binary contract")

    overall = summarize_group(judgments, ["run_id", "condition"])
    overall.insert(2, "category", "all")
    by_category = summarize_group(judgments, ["run_id", "condition", "category"])
    return pd.concat([overall, by_category], ignore_index=True)


def summarize_group(judgments, group_columns):
    summary = (
        judgments.groupby(group_columns, sort=True)["score"]
        .agg(rows="count", compliance_rate="mean")
        .reset_index()
    )
    summary["refusal_rate"] = 1.0 - summary["compliance_rate"]
    return summary
