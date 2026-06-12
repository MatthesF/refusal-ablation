import pytest

from src import dataset
from settings import POLICY_AREAS


def test_load_prompts_reports_missing_required_columns(tmp_path, monkeypatch):
    path = tmp_path / "prompts.csv"
    path.write_text("prompt,label\nhello,safe\n", encoding="utf-8")
    monkeypatch.setattr(dataset, "DATASET_PATH", path)

    with pytest.raises(ValueError, match="missing required columns: category"):
        dataset.load_prompts()


def test_load_prompts_rejects_empty_prompt_text(tmp_path, monkeypatch):
    rows = ["prompt,label,category"]
    for category in POLICY_AREAS:
        for label in ["safe", "unsafe"]:
            for prompt_index in range(20):
                prompt = f"{label} prompt {prompt_index} for {category}"
                if category == POLICY_AREAS[0] and label == "safe" and prompt_index == 0:
                    prompt = " "
                rows.append(f"\"{prompt}\",{label},{category}")

    path = tmp_path / "prompts.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(dataset, "DATASET_PATH", path)

    with pytest.raises(ValueError, match="dataset prompts must be non-empty"):
        dataset.load_prompts()


def test_policy_dataset_split_is_policy_area_disjoint(tmp_path, monkeypatch):
    rows = ["id,label,category,policy_area,prompt"]
    row_id = 1
    for category in POLICY_AREAS:
        for label in ["safe", "unsafe"]:
            for prompt_index in range(20):
                rows.append(
                    f"{row_id},{label},{category},{category},"
                    f"\"{label} prompt {prompt_index} for {category}\""
                )
                row_id += 1

    path = tmp_path / "gemma_policy_prompts.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(dataset, "DATASET_PATH", path)

    prompts = dataset.load_prompts()
    split = dataset.split_prompts(prompts)

    construction_categories = set(split.loc[split["split"] == "construction", "category"])
    test_categories = set(split.loc[split["split"] == "test", "category"])

    assert len(prompts) == 480
    assert len(construction_categories) == 8
    assert len(test_categories) == 4
    assert construction_categories.isdisjoint(test_categories)
