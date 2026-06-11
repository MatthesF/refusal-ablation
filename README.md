# Gemma Refusal Ablation

This repository runs one fixed experiment:

1. Fit a refusal direction from the local 480-prompt policy dataset.
2. Check whether the direction generalizes across held-out policy areas.
3. Generate SORRY-Bench answers with baseline Gemma.
4. Apply the refusal-direction edit and generate the same SORRY-Bench answers again.
5. Score both answer sets with the pinned official SORRY-Bench judge.

## RunPod Run

```bash
python -m pip install -r requirements.txt

python -m src.download_assets
python -m src.fit_direction
python -m src.run_gemma_sorry_bench

python -m src.download_official_judge
python -m pip install -r requirements-official-evaluator.txt
python -m src.score_sorry_bench_official
```

`src.download_assets` requires access to the gated
`sorry-bench/sorry-bench-202503` Hugging Face dataset and downloads the pinned
official evaluator code. Accept the dataset terms and export `HF_TOKEN` if
Hugging Face requires authentication.

`src.download_official_judge` downloads the gated official judge checkpoint. It
is needed for scoring only, so missing judge access does not block fitting the
direction or generating Gemma SORRY-Bench answers.

The official evaluator requirements are installed after Gemma generation so
vLLM's CUDA/Torch dependencies do not interfere with the base experiment
environment.

The RunPod runner installs the CUDA 12.8 PyTorch wheel explicitly because the
RTX 5090 pod driver reports CUDA 12.8; using a CUDA 13 wheel makes PyTorch unable
to see the GPU.

`requirements-official-evaluator.txt` pins the ordinary Python packages. vLLM is
left platform-resolved because its CPU and CUDA packages resolve differently;
the RunPod script records the exact installed vLLM version in the final
`pip freeze` artifact.

## Following A RunPod Run

Create or start the pod from the RunPod console or `runpodctl`, then SSH into
the pod and run the experiment from the repo root:

```bash
export HF_TOKEN=...
bash scripts/runpod_experiment.sh
```

The script writes a timestamped log in `artifacts/` and updates:

```text
artifacts/runpod_latest.log
artifacts/runpod_requirements_base_<timestamp>.txt
artifacts/runpod_requirements_official_<timestamp>.txt
```

If the SSH connection drops, reconnect and continue watching with:

```bash
tail -f artifacts/runpod_latest.log
```

## Inputs

Local direction-construction data:

```text
data/gemma_policy_prompts.csv
```

The GPT Pro prompt used to create the policy-derived construction dataset lives
in:

```text
prompts/gemma_policy_dataset_prompt.md
```

Required columns:

```text
prompt,label,category
```

Optional source-metadata columns such as `id` and `policy_area` may be present.
The loader creates its own stable runtime `id` after reading the file.

Labels must be exactly `safe` and `unsafe`. The dataset must contain the 12
Gemma policy areas in `category`, with exactly 20 safe and 20 unsafe prompts per
area. The split is policy-area-disjoint, so held-out local prompts come from
policy areas that were not used to fit the direction.

SORRY-Bench data lives locally under:

```text
data/sorry_bench/question.jsonl
```

This dataset is gated and is not committed.

## Outputs

```text
artifacts/prompt_split.csv
artifacts/direction_fit_report.csv
artifacts/category_generalization_table.csv
artifacts/refusal_directions.npz
artifacts/sorry_bench_outputs.csv
artifacts/sorry_bench_official_export_manifest.csv
artifacts/sorry_bench_official_judgments.csv
artifacts/sorry_bench_official_summary.csv
```

`sorry_bench_outputs.csv` contains raw baseline and edited Gemma answers.

`category_generalization_table.csv` checks whether the refusal direction is
shared across policy areas. For each category budget, it fits one direction on
the used policy areas and one direction on the remaining held-out policy areas,
then compares them layer by layer. The final edit is still fit only on the fixed
8 construction policy areas.

`sorry_bench_official_judgments.csv` contains one official judge score per
answer. The score contract is:

```text
0 = refusal
1 = unsafe compliance
```

`sorry_bench_official_summary.csv` reports compliance and refusal rates overall
and by SORRY-Bench category.

## Official Judge

The official evaluator scripts are vendored in:

```text
vendor/sorry_bench_official/
```

They are pinned to upstream commit:

```text
7da10addffb6790cfeb75281eaffb5a176861653
```

The downloader verifies SHA-256 hashes for every vendored official file. The
score step calls the official `gen_judgment_safety_vllm.py` script and passes
the vendored `judge_prompts.jsonl` explicitly. The judge prompt used is:

```text
base-ft-mistral-7b-instruct-v0.2
```

## Reproducibility

```text
Gemma model: google/gemma-4-E4B-it
Gemma revision: fee6332c1abaafb77f6f9624236c63aa2f1d0187
random seed: 2445
construction policy areas: 8
category generalization budgets: 2, 4, 6, 8 used policy areas
category generalization repeats: 5
category generalization reference: held-out policy-area direction
generation cap: 4096 new tokens
activation batch size: 32
generation batch size: 8
SORRY-Bench dataset: sorry-bench/sorry-bench-202503
SORRY-Bench dataset revision: 612a4e1f45db8adf884fa62318ddf9fa1c6e75e9
SORRY-Bench judge: sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406
SORRY-Bench judge revision: 79ab44668cef557414cb5e15c726a56ebca9cf1e
```

## File Roles

```text
settings.py                       fixed experiment constants
src/dataset.py                    local CSV loading and policy-area-disjoint split
src/gemma.py                      Gemma loading, generation, activations, edit
src/fit_direction.py              fit directions and write category generalization data
src/sorry_bench.py                SORRY-Bench loading, official export/import
src/download_assets.py            download dataset and official evaluator
src/download_official_judge.py    download gated official judge checkpoint
src/run_gemma_sorry_bench.py      generate baseline and edited Gemma answers
src/score_sorry_bench_official.py score answers with the official judge
scripts/runpod_experiment.sh      pod-side runner with live logs
```
