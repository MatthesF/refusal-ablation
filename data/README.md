# Local Data

This folder holds local inputs that are not downloaded from SORRY-Bench.

## Required Construction Dataset

Create this file before fitting the refusal direction:

```text
data/gemma_policy_prompts.csv
```

The generation prompt is saved at:

```text
prompts/gemma_policy_dataset_prompt.md
```

Columns read by the fitting code:

```text
prompt,label,category
```

The checked-in construction CSV also keeps source metadata:

```text
id,policy_area
```

Dataset contract:

```text
480 rows total
12 Gemma policy areas
20 safe prompts per policy area
20 unsafe prompts per policy area
category is one of the 12 stable policy ids
policy_area is the human-readable policy description
```

The fitting code validates this contract before computing activations. It creates
its own stable row id after loading, so the source `id` column is metadata rather
than part of the runtime contract.
