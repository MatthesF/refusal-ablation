# Results

Run date: 2026-06-10

Dataset:

- `/Users/matthesfogtmann/Desktop/statprojekt/project/experiments/teest/data/subcategory_balanced_prompts.csv`
- Construction activations:
  `/Users/matthesfogtmann/Desktop/statprojekt/project/experiments/arditi_minimal/artifacts/prompt_last_construction_hidden_states.npz`

## Raw Mean-Difference Direction

Configuration:

- Direction mode: `mean_diff`
- Rank: `1`
- Alpha: `1.0`
- Per-layer directions: yes
- Edit targets: `self_attn.o_proj`, `mlp.down_proj`
- Norm preservation: yes
- Validation subset: one prompt per subcategory, 10 safe + 10 unsafe

Result:

| Slice | Baseline refusal | Edited refusal |
|---|---:|---:|
| Safe | 0.00 | 1.00 |
| Unsafe | 1.00 | 1.00 |

Interpretation:

The raw mean-difference edit mechanically worked, but behaviorally failed. It
made safe prompts refuse and did not reduce unsafe refusals. This suggests the
raw unsafe-minus-safe direction removes benign answer capability or pushes the
model toward generic refusal.

## Biprojected Mean-Difference Direction

Configuration:

- Direction mode: `biprojected_mean_diff`
- Rank: `1`
- Alpha: `1.0`
- Per-layer directions: yes
- Edit targets: `self_attn.o_proj`, `mlp.down_proj`
- Norm preservation: yes
- Full validation split: 20 safe + 20 unsafe

Result:

| Slice | n pairs | Baseline refusal | Edited refusal | Difference | Discordant | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 40 | 0.50 | 0.00 | -0.50 | 20 | 1.91e-06 |
| Safe | 20 | 0.00 | 0.00 | 0.00 | 0 | 1.00 |
| Unsafe | 20 | 1.00 | 0.00 | -1.00 | 20 | 1.91e-06 |

Quality proxy:

| Slice | Refusal | Repetition | Very short | Cap hit | Median chars |
|---|---:|---:|---:|---:|---:|
| Baseline safe | 0.00 | 0.00 | 0.00 | 1.00 | 482.5 |
| Baseline unsafe | 1.00 | 0.00 | 0.20 | 0.05 | 73.5 |
| Edited safe | 0.00 | 0.00 | 0.00 | 1.00 | 496.0 |
| Edited unsafe | 0.00 | 0.00 | 0.00 | 1.00 | 508.0 |

Edit verification:

- Edited matrices: `84`
- Maximum relative norm error: `2.71e-07`
- Mean RMS direction coefficient before edit: `0.0241`
- Mean RMS direction coefficient after edit: `7.16e-09`

Interpretation:

The biprojected direction is the first simple recipe that worked on validation:
it suppressed unsafe refusals while preserving safe-prompt non-refusal and basic
quality proxies. The outputs still need manual review before final project
claims, especially because non-refusal does not imply useful or safe compliance.

Primary artifacts:

- Full validation summary:
  `/Users/matthesfogtmann/Desktop/statprojekt/project/experiments/norm_preserving_minimal/artifacts/validation_full_biprojected/SUMMARY.md`
- Full validation metrics:
  `/Users/matthesfogtmann/Desktop/statprojekt/project/experiments/norm_preserving_minimal/artifacts/validation_full_biprojected/summary.json`
- Full validation generations:
  `/Users/matthesfogtmann/Desktop/statprojekt/project/experiments/norm_preserving_minimal/artifacts/validation_full_biprojected/generations.csv`
