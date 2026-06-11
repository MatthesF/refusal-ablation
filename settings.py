from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data/gemma_policy_prompts.csv"
RUN_DIR = ROOT / "artifacts"

# Pinning the exact model revision makes the experiment reproducible.
GEMMA_MODEL_ID = "google/gemma-4-E4B-it"
GEMMA_MODEL_REVISION = "fee6332c1abaafb77f6f9624236c63aa2f1d0187"
RANDOM_SEED = 2445

# Policy-derived construction dataset contract.
POLICY_AREAS = [
    "intellectual_property_abuse",
    "illegal_goods_or_services",
    "crime_or_violence_facilitation",
    "extremist_or_terrorist_content",
    "unlicensed_professional_advice",
    "spam_fraud_phishing_or_malware",
    "safety_filter_circumvention",
    "hate_harassment_or_bullying",
    "self_harm_or_personal_harm",
    "privacy_surveillance_or_sensitive_data",
    "misinformation_impersonation_or_defamation",
    "sexual_content",
]
PROMPTS_PER_LABEL_PER_POLICY_AREA = 20
CONSTRUCTION_POLICY_AREAS = 8
DIRECTION_SPLIT_STYLE = "policy_area_disjoint"
CATEGORY_GENERALIZATION_POLICY_AREAS = [2, 4, 6, 8]
CATEGORY_GENERALIZATION_REPEATS = 5
CATEGORY_GENERALIZATION_REFERENCE = "held_out_policy_area_direction"
CATEGORY_GENERALIZATION_P10_QUANTILE = 0.10
GENERATION_MAX_NEW_TOKENS = 4096
ACTIVATION_BATCH_SIZE = 16
GENERATION_BATCH_SIZE = 4

# Files shared between fitting and output generation.
SPLIT_PATH = RUN_DIR / "prompt_split.csv"
POLICY_ACTIVATIONS_PATH = RUN_DIR / "policy_activations.npz"
DIRECTIONS_PATH = RUN_DIR / "refusal_directions.npz"
FIT_REPORT_PATH = RUN_DIR / "direction_fit_report.csv"
CATEGORY_GENERALIZATION_PATH = RUN_DIR / "category_generalization_table.csv"

# SORRY-Bench is gated on Hugging Face. The downloader expects the user to have
# accepted the dataset terms and exported HF_TOKEN when authentication is needed.
SORRY_BENCH_DATASET_ID = "sorry-bench/sorry-bench-202503"
SORRY_BENCH_DATASET_REVISION = "612a4e1f45db8adf884fa62318ddf9fa1c6e75e9"
SORRY_BENCH_DIR = ROOT / "data/sorry_bench"
SORRY_BENCH_QUESTION_PATH = SORRY_BENCH_DIR / "question.jsonl"
SORRY_BENCH_OUTPUTS_PATH = RUN_DIR / "sorry_bench_outputs.csv"

SORRY_BENCH_JUDGE_MODEL_ID = "sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406"
SORRY_BENCH_JUDGE_MODEL_REVISION = "79ab44668cef557414cb5e15c726a56ebca9cf1e"
SORRY_BENCH_JUDGE_MODEL_NAME = "ft-mistral-7b-instruct-v0.2"
SORRY_BENCH_OFFICIAL_JUDGE_PROMPT_NAME = "base-ft-mistral-7b-instruct-v0.2"
SORRY_BENCH_JUDGE_LOCAL_DIR = (
    ROOT / "ckpts/finetuned_models/ft-mistral-7b-instruct-v0.2-sorry-bench-202406"
)

SORRY_BENCH_OFFICIAL_COMMIT = "7da10addffb6790cfeb75281eaffb5a176861653"
SORRY_BENCH_OFFICIAL_DIR = ROOT / "vendor/sorry_bench_official"
SORRY_BENCH_OFFICIAL_EXPORT_MANIFEST_PATH = RUN_DIR / "sorry_bench_official_export_manifest.csv"
SORRY_BENCH_OFFICIAL_JUDGMENTS_PATH = RUN_DIR / "sorry_bench_official_judgments.csv"
SORRY_BENCH_OFFICIAL_SUMMARY_PATH = RUN_DIR / "sorry_bench_official_summary.csv"
