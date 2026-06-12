from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import gemma


def test_generation_room_reads_gemma4_text_context_window(monkeypatch):
    monkeypatch.setattr(gemma, "GENERATION_MAX_NEW_TOKENS", 4096)
    text_config = SimpleNamespace(max_position_embeddings=131_072)
    config = SimpleNamespace(
        get_text_config=lambda: text_config,
    )
    model = SimpleNamespace(config=config)

    assert gemma.generation_room(model, prompt_length=130_000) == 1072


def test_generation_room_rejects_prompts_outside_context_window():
    text_config = SimpleNamespace(max_position_embeddings=8)
    model = SimpleNamespace(config=SimpleNamespace(get_text_config=lambda: text_config))

    with pytest.raises(ValueError, match="longer than the model context window"):
        gemma.generation_room(model, prompt_length=8)


def test_last_non_padding_indices_handles_left_and_right_padding():
    attention_mask = torch.tensor([
        [0, 0, 1, 1],
        [1, 1, 1, 0],
    ])

    indices = gemma.last_non_padding_indices(attention_mask)

    assert indices.tolist() == [3, 2]


def test_refusal_directions_rejects_degenerate_activation_means():
    activations = np.zeros((2, 1, 3), dtype=np.float32)
    labels = np.array(["safe", "unsafe"])

    with pytest.raises(ValueError, match="safe activation mean contains zero vectors"):
        gemma.refusal_directions(activations, labels)


def test_remove_direction_rejects_zero_direction():
    weight = torch.eye(3)

    with pytest.raises(ValueError, match="refusal direction must be non-zero"):
        gemma.remove_direction(weight, torch.zeros(3))


def test_remove_direction_rejects_wrong_width():
    weight = torch.eye(3)

    with pytest.raises(ValueError, match="must match the Linear output dimension"):
        gemma.remove_direction(weight, torch.ones(2))


def test_remove_direction_preserves_column_norms():
    weight = torch.tensor([
        [1.0, 2.0, 1.0],
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 4.0],
    ])
    edited = gemma.remove_direction(weight, torch.tensor([1.0, 0.0, 0.0]))

    assert torch.allclose(
        torch.linalg.vector_norm(edited, dim=0),
        torch.linalg.vector_norm(weight, dim=0),
    )
    assert torch.allclose(edited[0], torch.zeros(3))


def test_remove_direction_rejects_columns_that_fully_collapse():
    weight = torch.eye(3)

    with pytest.raises(ValueError, match="cannot preserve norm"):
        gemma.remove_direction(weight, torch.tensor([1.0, 0.0, 0.0]))
