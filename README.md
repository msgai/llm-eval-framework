# llm-eval-framework

Standalone LLM agent evaluation framework — runs independently of `llm-service`.

## Project layout

```
llm-eval-framework/
├── agent_evaluation_script.py   # Main entry point (result gen + evaluation)
├── eval_utils.py                # All helpers: DMProxy factory, SOP/tool loaders
├── config.yaml                  # ⬅  ALL URLs, bot IDs, model names, paths here
├── .env                         # Secrets (gitignored) — OPENAI_API_KEY, ENV_CONFIG
├── requirements.txt
├── data/                        # Fallback static files
│   ├── tools_db.json
│   ├── sops.txt
│   ├── custom_tone_instructions.txt
│   └── adjust_response_rules.txt
├── prompts/
│   └── eval_prompt.txt
├── datasets/
│   └── llm_eval_dataset_1.pkl
└── results/                     # Created at runtime
```

## Setup

```bash
# 1. Create / activate your conda env
conda activate llm-service

# 2. Install dependencies (includes editable llm-service for DMProxy)
pip install -r requirements.txt

# 3. Copy the env template and fill in your key
cp .env .env.local          # or just edit .env directly
# set OPENAI_API_KEY and ENV_CONFIG (qa | dev)
```

## Running

To run the full pipeline (which fetches dynamic tone and rules, generates bot responses, and evaluates them):

```bash
python run_pipeline.py
```

Alternatively, you can run individual parts of the process:

* **Fetch dynamic instructions only**:
  ```bash
  python fetch_dynamic_instructions.py
  ```

* **Run evaluation script directly**:
  ```bash
  python agent_evaluation_script.py --model gpt-4o-123
  ```

* **Evaluation only (using pre-generated results)**:
  ```bash
  python agent_evaluation_script.py \
      --model gpt-4o-123 \
      --skip_generation \
      --results_file results/gpt-4o-123/agent_using_gpt-4o-123_FINAL.pkl
  ```


## Configuration

All tunable values live in **`config.yaml`** — edit it to switch environments,
change the bot under test, or point to a different LLM service endpoint.
Never hardcode URLs or credentials in Python source files.

| Section | Purpose |
|---|---|
| `bot` | Bot ID, ref ID, and target environment (e.g., qa) under evaluation |
| `api` | LLM-service endpoint and request parameters |
| `evaluation` | Judge model, batch size, reasoning effort |
| `config_service` | Netomi internal config service URLs (QA / DEV) |
| `dm_proxy_defaults` | Fallback DMProxy params if config service is unreachable |
| `dm_proxy_api` | `bot_env`, `capability_id`, `is_action_v2_enabled` |
| `prompts` / `config_data` | Relative paths to prompt and data files |
| `defaults` | CLI argument defaults |
