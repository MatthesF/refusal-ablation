from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data/safety_prompts_labeled.csv"
RUN_DIR = ROOT / "artifacts"

# Pinning the exact model revision makes the experiment reproducible.
MODEL_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fee6332c1abaafb77f6f9624236c63aa2f1d0187"
RANDOM_SEED = 2445

# Experiment choices that should match the report.
CONSTRUCTION_CATEGORIES_PER_LABEL = 8
STABILITY_FOLDS = 5

# Files shared between fitting and output generation.
SPLIT_PATH = RUN_DIR / "prompt_split.csv"
CONSTRUCTION_ACTIVATIONS_PATH = RUN_DIR / "construction_activations.npz"
DIRECTIONS_PATH = RUN_DIR / "refusal_directions.npz"
