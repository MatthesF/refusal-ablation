from __future__ import annotations

import torch

from run_validation import norm_preserving_output_projection


def main() -> None:
    torch.manual_seed(2445)
    weight = torch.randn(7, 11)
    direction = torch.randn(7)
    direction = direction / torch.linalg.vector_norm(direction)

    edited, stats = norm_preserving_output_projection(
        weight,
        direction,
        alpha=1.0,
        eps=1e-8,
    )

    before_norms = torch.linalg.vector_norm(weight, dim=0)
    after_norms = torch.linalg.vector_norm(edited, dim=0)
    residual = direction @ edited

    if not torch.allclose(before_norms, after_norms, atol=1e-5, rtol=1e-5):
        raise AssertionError("Column norms were not preserved")
    if not torch.allclose(residual, torch.zeros_like(residual), atol=1e-5, rtol=1e-5):
        raise AssertionError("Edited weight still writes into the removed direction")
    if stats["max_relative_norm_error"] > 1e-5:
        raise AssertionError(f"Unexpected norm error: {stats}")

    print("projection math ok")


if __name__ == "__main__":
    main()
