import json

import pandas as pd

from src import sorry_bench
from src.sorry_bench import (
    BASELINE_MODEL_ID,
    EDITED_MODEL_ID,
    export_official_answers,
    import_official_judgments,
    official_answer_dir,
    official_judge_prompt,
    official_judgment_path,
)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_official_judge_prompt_is_pinned():
    prompt = official_judge_prompt()

    assert prompt["name"] == "base-ft-mistral-7b-instruct-v0.2"
    assert "Output only the score." in prompt["prompt_template"]
    assert prompt["output_format"] == "scoreonly"


def test_export_official_answers_writes_baseline_and_edited_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(sorry_bench, "DIRECTIONS_PATH", tmp_path / "directions.npz")
    monkeypatch.setattr(sorry_bench, "SORRY_BENCH_DIR", tmp_path / "sorry_bench")
    monkeypatch.setattr(sorry_bench, "SORRY_BENCH_OUTPUTS_PATH", tmp_path / "outputs.csv")
    monkeypatch.setattr(sorry_bench, "SORRY_BENCH_OFFICIAL_EXPORT_MANIFEST_PATH", tmp_path / "manifest.csv")
    sorry_bench.DIRECTIONS_PATH.write_bytes(b"directions")
    run_id = sorry_bench.current_run_id()

    outputs = pd.DataFrame([
        {"run_id": run_id, "question_id": 2, "condition": "baseline", "answer": "baseline two"},
        {"run_id": run_id, "question_id": 1, "condition": "baseline", "answer": "baseline one"},
        {"run_id": run_id, "question_id": 1, "condition": "edited", "answer": "edited one"},
        {"run_id": run_id, "question_id": 2, "condition": "edited", "answer": "edited two"},
    ])
    outputs.to_csv(sorry_bench.SORRY_BENCH_OUTPUTS_PATH, index=False)

    manifest = export_official_answers()
    baseline_path = official_answer_dir() / f"{BASELINE_MODEL_ID}.jsonl"
    edited_path = official_answer_dir() / f"{EDITED_MODEL_ID}.jsonl"
    baseline_records = [json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines()]
    edited_records = [json.loads(line) for line in edited_path.read_text(encoding="utf-8").splitlines()]

    assert manifest[["condition", "model_id", "rows"]].to_dict("records") == [
        {"condition": "baseline", "model_id": BASELINE_MODEL_ID, "rows": 2},
        {"condition": "edited", "model_id": EDITED_MODEL_ID, "rows": 2},
    ]
    assert [record["question_id"] for record in baseline_records] == [1, 2]
    assert baseline_records[0]["choices"] == [{"index": 0, "turns": ["baseline one"]}]
    assert edited_records[1]["choices"] == [{"index": 0, "turns": ["edited two"]}]


def test_import_official_judgments_maps_scores_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(sorry_bench, "SORRY_BENCH_DIR", tmp_path / "sorry_bench")
    monkeypatch.setattr(sorry_bench, "SORRY_BENCH_QUESTION_PATH", tmp_path / "sorry_bench/question.jsonl")
    monkeypatch.setattr(sorry_bench, "SORRY_BENCH_OFFICIAL_JUDGMENTS_PATH", tmp_path / "judgments.csv")
    monkeypatch.setattr(sorry_bench, "SORRY_BENCH_OFFICIAL_SUMMARY_PATH", tmp_path / "summary.csv")

    write_jsonl(sorry_bench.SORRY_BENCH_QUESTION_PATH, [
        {"question_id": 1, "category": "11", "turns": ["question one"]},
        {"question_id": 2, "category": "12", "turns": ["question two"]},
    ])
    write_jsonl(official_judgment_path(), [
        {"question_id": 1, "model": EDITED_MODEL_ID, "score": 1, "judgment": "1", "tstamp": 2},
        {"question_id": 2, "model": EDITED_MODEL_ID, "score": 0, "judgment": "0", "tstamp": 3},
    ])

    manifest = pd.DataFrame([{
        "run_id": "run-1",
        "condition": "edited",
        "model_id": EDITED_MODEL_ID,
        "answer_file": "unused.jsonl",
        "rows": 2,
    }])

    judgments, summary = import_official_judgments(manifest)

    assert judgments[["question_id", "category", "score", "score_label"]].to_dict("records") == [
        {"question_id": 1, "category": "11", "score": 1, "score_label": "unsafe_compliance"},
        {"question_id": 2, "category": "12", "score": 0, "score_label": "refusal"},
    ]
    overall = summary.loc[summary["category"] == "all"].iloc[0]
    assert overall["rows"] == 2
    assert overall["compliance_rate"] == 0.5
