import numpy as np

from src.fit_direction import category_generalization_table


def test_category_generalization_table_compares_used_categories_to_held_out_categories():
    labels = []
    categories = []
    activations = []

    for category_index in range(12):
        category = f"policy-area-{category_index}"
        for label, base in [("safe", [0.0, 1.0, 0.0]), ("unsafe", [1.0, 0.0, 0.0])]:
            for prompt_index in range(2):
                labels.append(label)
                categories.append(category)
                prompt_offset = 0.001 * prompt_index
                activations.append([
                    [base[0], base[1], prompt_offset],
                    [base[0], base[1], -prompt_offset],
                ])

    table = category_generalization_table(
        np.array(activations, dtype=np.float32),
        np.array(labels),
        np.array(categories),
    )

    assert table["used_policy_areas"].tolist() == [2, 4, 6, 8]
    assert table["held_out_policy_areas"].tolist() == [10, 8, 6, 4]
    assert table["used_prompts"].tolist() == [8, 16, 24, 32]
    assert table["held_out_prompts"].tolist() == [40, 32, 24, 16]
    assert np.allclose(table["median_layer_cosine_to_heldout"], 1.0)
    assert np.allclose(table["p10_layer_cosine_to_heldout"], 1.0)
