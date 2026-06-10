# Minimal Norm-Preserving Refusal Edit

This folder tests the simplest Gemma-4-E4B refusal-edit recipe that is still close
to public Gemma abliteration work:

1. Use construction activations only.
2. For each language layer, compute one direction:

   ```text
   refusal_direction[layer] = mean(unsafe activations) - mean(safe activations)
   ```

3. Optionally biproject the direction by removing the part parallel to the safe
   activation mean. This is the simplest public-style guardrail against damaging
   harmless behavior.

4. Edit the two modules that write back into the residual stream:

   ```text
   self_attn.o_proj
   mlp.down_proj
   ```

5. For each PyTorch linear weight `W` with shape `[out_features, in_features]`,
   remove the output-space refusal direction from every column:

   ```text
   W_removed = W - alpha * r (r^T W)
   ```

6. Preserve each column norm:

   ```text
   W_new[:, j] = W_removed[:, j] * ||W[:, j]|| / ||W_removed[:, j]||
   ```

The default test uses `alpha = 1`, rank 1, and one direction per layer. It is
intended as a readable validation experiment, not a tuned abliteration system.

Run the math test:

```bash
/Users/matthesfogtmann/Desktop/statprojekt/project/experiments/teest/.venv-gemma4/bin/python \
  scripts/test_projection_math.py
```

Run the validation pilot:

```bash
/Users/matthesfogtmann/Desktop/statprojekt/project/experiments/teest/.venv-gemma4/bin/python \
  scripts/run_validation.py \
  --one-per-subcategory \
  --direction-mode biprojected_mean_diff
```

Outputs are written under `artifacts/`.
