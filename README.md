# Contract Review Agent — Prototype

AI-based orchestrator-workers agent for contract review, built for the
Summer School "Francesco Turco" 2026 paper.

## Quick start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key (create a .env file in the root)
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 4. Test the connection
python test_connection.py

# 5. Download a small sample of CUAD
python data/prepare_cuad.py

# 6. Run a single contract through the orchestrator (smoke test)
python experiments/smoke_test.py

# 7. Run full experiments (3 conditions)
python experiments/run_all.py

# 8. Evaluate results
python experiments/evaluate.py
```

## Project structure

```
contract-agent/
├── .env                       # API key (DO NOT COMMIT)
├── requirements.txt
├── config.py                  # central config
├── llm_client.py              # thin wrapper around Anthropic API
├── test_connection.py         # smoke test for the API
├── agents/
│   ├── orchestrator.py        # coordinates the workers
│   ├── segmenter.py           # splits contract into chunks
│   ├── extractor.py           # finds clauses
│   ├── classifier.py          # assigns CUAD labels
│   └── risk_scorer.py         # flags risk level
├── validators/
│   └── rules.py               # rule-based validation layer
├── data/
│   ├── categories.json        # the 10 clause categories we focus on
│   └── prepare_cuad.py        # downloads CUAD and saves a sample
├── experiments/
│   ├── smoke_test.py          # single contract run
│   ├── run_all.py             # runs all 3 conditions
│   └── evaluate.py            # computes precision / recall / F1
└── results/
    └── (JSON output files)
```

## Architecture

Orchestrator-workers pattern (Anthropic 2024):

1. The orchestrator receives a contract
2. It plans which categories to look for (dynamic task decomposition)
3. It segments the contract (chunking)
4. It dispatches extraction + classification + risk scoring to workers
5. It aggregates results and sends them to the rule-based validator
6. It returns structured output for human review

## Three experimental conditions

- **A: LLM zero-shot** — single prompt, no decomposition, no validation
- **B: Multi-agent orchestrator** — decomposition and workers, no validation
- **C: Full system** — orchestrator + rule-based validation layer

Comparing A, B, C gives an ablation study: how much does each design
choice contribute to the final performance?
