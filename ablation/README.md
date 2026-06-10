# Ablation Experiment

This folder contains the small Gemma ablation experiment used for the project.

The idea:

1. Split the prompt dataset into construction prompts and held-out test prompts.
2. Keep four unsafe categories completely out of construction.
3. Fit one refusal direction per layer from 50 safe and 50 unsafe construction prompts.
4. Report 5-fold stability of the 100-prompt direction.
5. Generate baseline and edited model outputs on the held-out test prompts.
6. Manually label the paired outputs as `accepted`, `refused`, or `unusable`.

Run:

```bash
python ablation/fit_direction.py
python ablation/run_outputs.py
```

Input data must be here:

```text
ablation/data/safety_prompts_labeled.csv
```

Required columns:

```text
prompt,label,category
```

Important outputs:

```text
ablation/artifacts/prompt_split.csv
ablation/artifacts/direction_fit_report.csv
ablation/artifacts/refusal_directions.npz
ablation/artifacts/baseline_outputs.csv
ablation/artifacts/edited_outputs.csv
ablation/artifacts/evaluation_outputs.csv
```

`refusal_directions.npz` also stores the model revision, fit prompt ids, K-fold
cosine numbers, and held-out unsafe categories.

`run_outputs.py` writes baseline and edited outputs separately while it runs, so
an interrupted run can continue without starting from zero.

Method source:

The direction fit follows the refusal-direction idea from Arditi et al.,
[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717):
estimate a direction from unsafe minus safe activations, then remove that
direction from model weights.

The extra projection against the safe mean and the norm-preserving edit follow
the simpler Gemma-style recipe used in
[TrevorS/gemma-4-abliteration](https://github.com/TrevorS/gemma-4-abliteration).

Reproducibility:

```text
model: google/gemma-4-E4B-it
revision: fee6332c1abaafb77f6f9624236c63aa2f1d0187
random seed: 2445
dataset sha256: d080e6c958ab9b17d99d99f02cdd708726b1c2ecf92b541e6f31f4c9a7ec7519
construction: 540 prompts
test: 460 prompts
test groups: 200 safe, 160 seen_unsafe, 100 heldout_unsafe
```

File roles:

```text
settings.py        fixed paths and experiment choices
dataset.py         pandas dataset loading and split
model.py           Gemma loading, activations, generation, weight edit
fit_direction.py   split data, report K-fold cosine, fit directions
run_outputs.py     generate baseline and edited outputs
```
