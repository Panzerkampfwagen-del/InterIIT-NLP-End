# Agentic Customer 360 - Proactive Intervention Desk

A streaming, multi-agent, human-in-the-loop decision system that watches customer
signals (transactions, tickets, usage, KYC) as they arrive, keeps a live structured
picture of each customer, infers life events / risks / opportunities, and makes a
bounded, explainable decision - an explicit **action** or an explicit **no-action**,
never silence - wrapped in deterministic guardrails with full traceability.

> This is **not** a chatbot and **not** a dashboard-with-agents-attached. The system's
> reason for existing is the decision boundary: `ACTION | NO_ACTION`, evidenced and
> confidence-scored.

## Architecture

```
Live/Replayed Event → Customer Identification → Streaming Processing (watermarks)
    → Specialized Signal Agents (parallel) → Shared Customer State Board
    → Cross-Agent Correlation (trigger evaluator) → Life-Event/Risk/Opportunity Inference
    → Action Decision → Policy/Guardrails (deterministic, post-proposal)
    → HITL when required (durable, blocking) → Final Action OR Explicit No-Action
    → Evidence + Explanation + Trace → Evaluation Output (inferred-events checkpoints)
```

Stack: Redpanda (Kafka API) · custom event-time-aware consumer with watermark table
(Flink semantics, no Flink) · PostgreSQL (state board, episodic store, identity graph,
pgvector + BM25 hybrid retrieval) · Redis (hot reads) · typed state-graph orchestration
· Groq LLM for synthesis only (statistical-first detection) · OpenTelemetry GenAI spans
· deterministic policy state machine · RAGAS-style decomposed evaluation harness.

## Repository layout

```
src/
  ingestion/    event schemas, dataset loader, replay simulator (SIMULATION, separated)
  identity/     deterministic + probabilistic resolver, identity graph
  streaming/    watermark-aware consumer, windowed feature computation
  state/        customer state board, episodic store, decay, conflict log
  memory/       working/episodic/semantic access layers
  retrieval/    hybrid retrieval, incremental indexer, embeddings
  agents/       signal agents (usage/transaction/support/kyc), life-event, action, critique
  orchestration/ typed state graph, trigger evaluator
  action/       action selection engine + eligibility
  guardrails/   deterministic policy state machine (post-proposal, fail-closed)
  hitl/         durable pending_approval store + transitions
  explainability/ deterministic explain() from stored evidence
  observability/  OTel GenAI spans, context propagation, PII scanner
  security/     RBAC, tokenization, retrieval isolation enforcement
evaluation/     scenarios, ground truth, inferred_events output, harness, dataset
scripts/        run_demo.py, hitl.py, run_pipeline.py, run_evaluation.py
tests/          unit / integration / scenario
```

## How to run

```bash
# 1. Infrastructure (Redpanda + Postgres/pgvector + Redis + OTel collector)
docker compose up -d

# 2. Python environment
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 3. Initialize the database schema
.venv/bin/python scripts/init_db.py

# 4. Run one scenario end-to-end (replay simulator → pipeline → inferred-events)
.venv/bin/python scripts/run_demo.py --scenario scenario_01

# 5. Score outputs against the provided ground truth
.venv/bin/python scripts/run_evaluation.py

# 6. Tests
.venv/bin/python -m pytest
```

Configuration is environment-driven: `C360_*` variables (see `src/config.py`); the
LLM provider reads `GROQ_API_KEY` from the environment. All LLM-dependent steps have
a deterministic fallback contract - the pipeline must never require a live LLM to
produce a valid decision object.