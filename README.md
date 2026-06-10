# Norm-Preserving Refusal Edit for Gemma 4 E4B

This project tests whether a small, open-weight Gemma 4 model's refusal behavior
can be changed by editing model weights along a measured refusal direction.

The working experiment is the norm-preserving edit in
`project/experiments/norm_preserving_minimal/`. This is the version that
currently produces a clear behavioral effect in validation.

## Model

The current model is:

- `google/gemma-4-E4B-it`

This is the instruction-tuned Gemma 4 edge model. It is small enough to run
locally on the available Apple Silicon setup, while still giving access to the
internal weights and activations needed for a white-box refusal edit.

The relevant architecture detail for this project is that each language layer
writes back into the residual stream through projection matrices. The working
edit targets the two modules that directly write layer outputs back into that
stream:

- `self_attn.o_proj`
- `mlp.down_proj`

For each targeted PyTorch linear weight `W` with shape
`[out_features, in_features]`, the edit removes the measured refusal direction
from the output space of every column, then rescales each column back to its
original norm.

## Research Question

The project asks:

> Can a norm-preserving refusal-direction edit reduce unsafe-prompt refusals in
> Gemma 4 E4B without causing safe prompts to be refused?

This is a safety evaluation. The goal is to measure whether refusal behavior is
robust to a simple white-box model edit, not to publish harmful outputs or build
a jailbreak tool.

## Data

The evaluation uses a balanced prompt dataset with safe and unsafe prompts.

Safe prompts are requests that a helpful model should answer. Unsafe prompts are
requests that a responsible model should refuse or redirect. Prompts are split
by role so the same data is not used for every part of the experiment:

- construction prompts estimate the refusal direction;
- validation prompts test whether the edit changes behavior;
- held-out prompts should remain untouched until the method is frozen.

Unsafe prompt text and unsafe generations should be handled as private
evaluation material. Public writeups should use aggregate rates, labels, and
sanitized examples.

## Method

The working edit is a biprojected mean-difference direction with norm
preservation.

First, construction activations are collected for safe and unsafe prompts. For
each layer, the base direction is:

```text
mean(unsafe activations) - mean(safe activations)
```

The working version then removes the part of this direction that is parallel to
the safe activation mean. This is the "biprojected" step. It is intended to make
the edit less destructive to normal safe-prompt behavior.

For every targeted projection matrix, the edit removes the refusal direction:

```text
W_removed = W - alpha * r (r^T W)
```

Then it restores the original norm of each column:

```text
W_new[:, j] = W_removed[:, j] * ||W[:, j]|| / ||W_removed[:, j]||
```

Current working configuration:

- Direction mode: `biprojected_mean_diff`
- Rank: `1`
- Alpha: `1.0`
- Per-layer directions: yes
- Edit targets: `self_attn.o_proj`, `mlp.down_proj`
- Norm preservation: yes
- Decoding temperature: `0.0`
- Max new tokens: `96`

## Current Validation Result

The current validation run uses 40 paired prompts:

- 20 safe prompts
- 20 unsafe prompts

Heuristic refusal labels show:

| Slice | n pairs | Baseline refusal | Edited refusal | Difference | Discordant | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 40 | 0.50 | 0.00 | -0.50 | 20 | 1.91e-06 |
| Safe | 20 | 0.00 | 0.00 | 0.00 | 0 | 1.00 |
| Unsafe | 20 | 1.00 | 0.00 | -1.00 | 20 | 1.91e-06 |

The edit verification also passed mechanically:

- Edited matrices: `84`
- Maximum relative norm error: `2.71e-07`
- Mean RMS direction coefficient before edit: `0.0241`
- Mean RMS direction coefficient after edit: `7.16e-09`

Plain interpretation:

> In the current validation run, the biprojected norm-preserving edit removed
> the measured refusal-direction component from the targeted matrices and
> suppressed unsafe-prompt refusals without increasing safe-prompt refusals.

This is not yet a final project claim. The current labels are heuristic
debugging evidence. Manual review is still required before reporting final
statistical claims.

## What This Does And Does Not Show

This result supports a narrow claim:

- Gemma 4 E4B refusal behavior can be strongly changed by a simple
  norm-preserving direction edit on this validation set.

It does not prove:

- that the model gives useful harmful instructions after editing;
- that all refusal behavior is controlled by one true direction;
- that the same edit works across all prompts, models, or decoding settings.

For the report, refusal suppression should be separated from harmful compliance
and output quality.

## Project Files

- `project/experiments/norm_preserving_minimal/README.md`: method notes and run
  commands.
- `project/experiments/norm_preserving_minimal/RESULTS.md`: current validation
  results.
- `project/experiments/norm_preserving_minimal/SANITIZED_EXAMPLES.md`: safe
  examples and redacted unsafe behavior summary.
- `project/experiments/norm_preserving_minimal/scripts/run_validation.py`: main
  validation script.
- `project/experiments/norm_preserving_minimal/scripts/run_paired_full_outputs.py`:
  paired baseline-versus-edited output generation.
- `project/experiments/norm_preserving_minimal/scripts/test_projection_math.py`:
  math check for the norm-preserving projection.

## Responsible Reporting

When presenting or publishing this project:

- describe it as a controlled safety evaluation;
- report refusal rates and paired statistical tests;
- use sanitized examples;
- do not publish raw unsafe generations;
- state that the current evidence is validation evidence until manual review is
  complete;
- state that the result is specific to Gemma 4 E4B, this dataset, and this edit
  implementation.
