"""Model identifiers, conditions, paths and repetition counts.

Model identifiers are authoritative: never edit, correct or substitute one,
even if it looks unfamiliar.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

CORPUS_PATH = DATA_DIR / "texts.tsv"
PILOT_CORPUS_PATH = DATA_DIR / "pilot_texts.tsv"
GOLD_PATH = DATA_DIR / "results_new.xlsx"

# Inputs and outputs of prep_corpus.py. The .docx sources hold one learner text
# per table row; set these to your own filenames or pass --sources. The name map
# is the anonymisation table and is never committed.
CORPUS_SOURCE_FILES = [
    DATA_DIR / "dataset_1.docx",
    DATA_DIR / "dataset_2.docx",
]
CORPUS_ORIGINAL_PATH = DATA_DIR / "texts_original.tsv"
NAME_MAP_PATH = DATA_DIR / "name_map.json"

# Each format maps to a prompt template in prompts/.
OUTPUT_FORMATS = ["lists", "counts"]

REPETITIONS = 3

# Every provider is called through the OpenAI SDK; DashScope and Mistral
# expose OpenAI-compatible endpoints. base_url None means the SDK default.
PROVIDERS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    },
    "mistral": {
        "api_key_env": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
    },
}

# One entry per experimental condition. Reasoning on and off share a model_id
# and differ only in how the provider is asked to switch reasoning: GPT and
# Mistral take reasoning_effort as a request parameter, DashScope takes
# enable_thinking in extra_body.
#
# mistral-medium-2604 is the date-pinned identifier for Medium 3.5. Phase 2 was
# run under the undated alias mistral-medium-3-5, which resolved to this same
# build; the alias may move to a future release.
_CONDITIONS = [
    ("gpt-5.4-nano_reasoning-off", "gpt-5.4-nano-2026-03-17", "openai",
     {"reasoning_effort": "none"}, None),
    ("gpt-5.4-nano_reasoning-on", "gpt-5.4-nano-2026-03-17", "openai",
     {"reasoning_effort": "high"}, None),
    ("qwen-3.5-flash_reasoning-off", "qwen3.5-flash-2026-02-23", "dashscope",
     {}, {"enable_thinking": False}),
    ("qwen-3.5-flash_reasoning-on", "qwen3.5-flash-2026-02-23", "dashscope",
     {}, {"enable_thinking": True}),
    ("gpt-5.4_reasoning-off", "gpt-5.4-2026-03-05", "openai",
     {"reasoning_effort": "none"}, None),
    ("gpt-5.4_reasoning-on", "gpt-5.4-2026-03-05", "openai",
     {"reasoning_effort": "high"}, None),
    ("qwen-3.5-plus_reasoning-off", "qwen3.5-plus-2026-02-15", "dashscope",
     {}, {"enable_thinking": False}),
    ("qwen-3.5-plus_reasoning-on", "qwen3.5-plus-2026-02-15", "dashscope",
     {}, {"enable_thinking": True}),
    ("mistral-small_reasoning-off", "mistral-small-2603", "mistral",
     {"reasoning_effort": "none"}, None),
    ("mistral-small_reasoning-on", "mistral-small-2603", "mistral",
     {"reasoning_effort": "high"}, None),
    ("mistral-medium_reasoning-off", "mistral-medium-2604", "mistral",
     {"reasoning_effort": "none"}, None),
    ("mistral-medium_reasoning-on", "mistral-medium-2604", "mistral",
     {"reasoning_effort": "high"}, None),
]

MODELS = [
    {
        "name": name,
        "model_id": model_id,
        "provider": provider,
        "params": params,
        **({"extra_body": extra_body} if extra_body else {}),
        **PROVIDERS[provider],
    }
    for name, model_id, provider, params, extra_body in _CONDITIONS
]

# USD per million tokens, keyed by model_id. Used by extract_results.py to
# estimate cost per call retroactively; a model_id that is absent gives a
# blank cost column. mistral-medium-3-5 is the alias phase 2 was called with,
# kept so records saved before the identifier was pinned still resolve.
PRICING = {
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "qwen3.5-flash-2026-02-23": {"input": 0.10, "output": 0.40},
    "qwen3.5-plus-2026-02-15": {"input": 0.40, "output": 2.40},
    "mistral-small-2603": {"input": 0.15, "output": 0.60},
    "mistral-medium-2604": {"input": 1.50, "output": 7.50},
    "mistral-medium-3-5": {"input": 1.50, "output": 7.50},
}

# Corpus, conditions and repetitions for one batch of the experiment, selected
# with --phase. Individual CLI flags override these. To run a phase in
# parallel, start several processes with different --models on the same phase.
PHASES = {
    "pilot": {
        "corpus": PILOT_CORPUS_PATH,
        "models": [
            "gpt-5.4-nano_reasoning-off",
            "gpt-5.4-nano_reasoning-on",
            "qwen-3.5-flash_reasoning-off",
            "qwen-3.5-flash_reasoning-on",
            "mistral-small_reasoning-off",
            "mistral-small_reasoning-on",
        ],
        "reps": 1,
    },
    "phase1": {
        "corpus": CORPUS_PATH,
        "models": [
            "gpt-5.4-nano_reasoning-off",
            "gpt-5.4-nano_reasoning-on",
            "qwen-3.5-flash_reasoning-off",
            "qwen-3.5-flash_reasoning-on",
            "mistral-small_reasoning-off",
            "mistral-small_reasoning-on",
        ],
        "reps": 3,
    },
    "phase2": {
        "corpus": CORPUS_PATH,
        "models": [
            "gpt-5.4_reasoning-off",
            "gpt-5.4_reasoning-on",
            "qwen-3.5-plus_reasoning-off",
            "qwen-3.5-plus_reasoning-on",
            "mistral-medium_reasoning-off",
            "mistral-medium_reasoning-on",
        ],
        "reps": 3,
    },
}
