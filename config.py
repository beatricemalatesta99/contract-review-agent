"""
Central configuration for the project.

Keeping config in one place means you change model, paths, or sample size
in a single file, and every experiment uses the same settings. This also
helps with reproducibility — you can cite "config.py commit abc123" in
the paper.
"""

from pathlib import Path

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CUAD_SAMPLE_PATH = DATA_DIR / "cuad_sample.json"
CATEGORIES_PATH = DATA_DIR / "categories.json"

# -------------------------------------------------------------------
# LLM configuration
# -------------------------------------------------------------------
# Claude Sonnet 4.5 — good balance of quality, cost, and context length.
# For a quick sanity check you can also use Haiku (cheaper, faster,
# slightly less accurate):
#   MODEL = "claude-haiku-4-5"
MODEL = "claude-sonnet-4-5-20250929"

# Deterministic output for reproducibility in experiments.
# Non-zero temperature is fine for exploration, but for the paper you
# want results that are reproducible.
TEMPERATURE = 0.0

# Upper bound on output tokens per call. 4000 is enough for most clause
# extraction / classification outputs. Adjust if you see truncation.
MAX_OUTPUT_TOKENS = 4000

# -------------------------------------------------------------------
# Experiment configuration
# -------------------------------------------------------------------
# Start small, scale up once you know the pipeline works end-to-end.
# 10 contracts is enough for a smoke test. Move to 30–50 for the real
# experiment. Going to 100+ eats budget without much added statistical
# power for a workshop paper.
N_CONTRACTS = 10

# Retry on API failures (rate limits, transient errors).
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 5
