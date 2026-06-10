# Ablation Experiment

This repository contains the Gemma refusal-ablation experiment used for the
project.

The idea:

1. Split the dataset by category, not by prompt.
2. Use 8 safe and 8 unsafe categories for direction construction.
3. Keep every remaining category for evaluation.
4. Fit one refusal direction per layer from all construction prompts.
5. Check direction stability with 5 independent category-stratified prompt folds.
6. Generate baseline and edited model outputs on the category-disjoint evaluation set.
7. Manually label paired outputs as `accepted`, `refused`, or `unusable`.

Run:

```bash
python -m src.fit_direction
python -m src.run_outputs
```

Input data must be here:

```text
data/safety_prompts_labeled.csv
```

Required columns:

```text
prompt,label,category
```

Current split:

```text
total prompts: 1000
construction: 400 prompts from 16 categories
evaluation: 600 prompts from 24 categories
evaluation groups: 300 safe, 300 unsafe
```

The category split uses `sklearn.model_selection.train_test_split`, stratified by
the category label. With the current seed, construction categories are:

```text
safe:
- Emergency preparedness
- Fraud and scam prevention
- General health and medical safety
- Legal process literacy
- Model safety evaluation
- Personal safety planning
- Product trust and safety
- Self-harm prevention and support

unsafe:
- AI-enabled impersonation and abuse
- Academic and professional fraud
- Chemical, biological, and poisoning harm
- Critical infrastructure sabotage
- Fraud, theft, and deception
- Hate, harassment, and extremism
- Malware and unauthorized intrusion
- Public deception and incitement
```

Important outputs:

```text
artifacts/prompt_split.csv
artifacts/direction_fit_report.csv
artifacts/refusal_directions.npz
artifacts/baseline_outputs.csv
artifacts/edited_outputs.csv
artifacts/evaluation_outputs.csv
```

`refusal_directions.npz` stores the model revision, construction prompt ids,
construction categories, and fold-stability cosine numbers.

`run_outputs.py` writes baseline and edited outputs separately while it runs, so
an interrupted run can continue. Output rows include a run id based on the fitted
direction file and token limit, which prevents stale outputs from being reused
after the experiment settings change.

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
construction categories per label: 8
stability folds: 5
max new tokens: 256
```

File roles:

```text
settings.py          fixed paths and experiment choices
src/dataset.py       pandas dataset loading and category-disjoint split
src/model.py         Gemma loading, activations, generation, weight edit
src/fit_direction.py split data, report fold stability, fit directions
src/run_outputs.py   generate baseline and edited outputs
```
