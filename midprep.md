# Agentic Customer 360
## Submission: Architecture + Research + Implementation Plan

# PART 1 : ARCHITECTURE

## 1.1 The full intended path

```
Live / Replayed Event (a replay simulator, clearly labeled as simulation)
  → Customer Identification (deterministic-first matching, probabilistic fallback, identity graph)
  → Streaming Processing (event-time watermarks, deduplication, windowed baselines)
  → Specialized Signal Agents (parallel fan-in: usage, transactions, support, KYC)
  → Shared Customer State Board (versioned state, episodic store, conflict log)
  → Cross-Agent Correlation (trigger evaluator, cost-disciplined)
  → Life-Event / Risk / Opportunity Inference (structured LLM synthesis + deterministic fallback)
  → Critique / Arbitration (conflict resolution, losing values kept for audit)
  → Action Decision (bounded action space, deterministic eligibility)
  → Policy / Guardrails (hard rules evaluated after the proposal, fail-closed)
  → HITL when required (durable pending-approval, physically blocking)
  → Final Action OR Explicit No-Action (never silence)
  → Evidence + Explanation + Trace (deterministic decision log, zero LLM)
  → Evaluation Output (staged decomposition + checkpoint scoring)
```

## 1.2 Stack to be used 

| Layer | Choice | Rejected alternatives |
|---|---|---|
| Event bus | **Redpanda** (Kafka-API compatible) | Kafka (fine if present), Pulsar (operational overhead), Redis Streams (weaker durability/replay) |
| Stream processing | **Event-time-aware consumer + watermark table** (Flink *semantics*, no Flink , scale-conditioned) | Full Flink/Dataflow cluster (disproportionate at project scale) |
| State / memory | **PostgreSQL** (JSONB + relational integrity, one instance serves state + episodic + identity + retrieval) | Redis as source of truth (hot cache only), dedicated vector DB, temporal-KG store |
| Retrieval | **pgvector + Postgres full-text search (BM25), hybrid, recency re-ranked** (hybrid similarity score minus a per-day age penalty) | Vector-only, GraphRAG (corpus lacks multi-hop relational structure) |
| Embeddings | **Deterministic local hashed-n-gram vectors** (256-dimensional, L2-normalized) | External embeddings API (heavy dependency footprint) |
| Orchestration | **Typed state-graph design** (explicit nodes, persisted checkpoints, replayable traces) | CrewAI role-DSL, AutoGen conversation history |
| LLM | **Groq (OpenAI-compatible endpoint), synthesis only** - statistical-first detection | LLM as primary detector (explicitly rejected per PS §4.6); LLM-driven memory paging |
| Observability | **OpenTelemetry GenAI semantic conventions** plus custom attributes, 100% sampled | Ad-hoc logging, raw chain-of-thought in spans |
| Guardrails | **Deterministic policy state machine**, evaluated AFTER the agent proposal | Prompt-layer policy instructions (bypassable) |
| Evaluation | **RAGAS-style decomposed harness** + dataset checkpoint scoring | Single blended accuracy score |

## 1.3 Non-negotiable components (Production Bar)

| Requirement | Design approach |
|---|---|
| Explicit action-or-no-action, never silence | every decision cycle resolves to a bounded action or an explicit no-action |
| Genuine streaming (not batch file processing) | an event bus plus a replay simulator clearly labeled as simulation |
| Event-time / out-of-order / late / duplicate handling | watermark tracking, allowed-lateness recomputation, per-customer deduplication |
| Identity isolation (no cross-customer contamination) | deterministic-first matching → probabilistic fallback → identity graph; retrieval always scoped to a single customer |
| Structured, versioned, provenanced state | compare-and-swap versioning + episodic supersession + conflict log + lazy decay |
| Real multi-agent disagreement + arbitration | a dedicated conflict log keeping both values, never deleted, + an arbitration step |
| Guardrails immune to prompt persuasion | deterministic hard rules evaluated after the agent proposal; fail-closed on internal error |
| Durable, blocking HITL | a pending-approval queue in Postgres that survives restarts; approve / reject / modify |
| Explainability without post-hoc reconstruction | deterministic assembly from decision-log snapshots; zero LLM |
| Full traceability | OpenTelemetry GenAI spans + W3C trace context propagated across the bus boundary |
| PII protection / retrieval isolation | structurally-enforced customer scoping (an unscoped query is not expressible) + PII scanners |
| Evaluation against hidden ground truth | staged decomposition + checkpoint scoring on the dataset's fixed enums |
| Reproducible decision logs | versioned state + episodic reconstruction at any past point |

## 1.4 Architectural novelty

1. **Trigger-driven agent invocation** : the life-event agent fires only on correlation-worthy
   conditions (multiple finding updates in a window / high severity / a KYC hard flag) with a
   cooldown; never per-event. Reasoning is worth its cost only when correlation is possible.
2. **A deliberately minimal agent roster** : four signal agents + one inference agent; KYC is a
   fully deterministic rule engine with **zero LLM calls**; every LLM role is explicit.
3. **MemGPT/CoALA memory vocabulary applied at the state-board level** : externally-controlled
   hot/cold paging (bounded current state vs episodic history), not per-agent self-paging;
   narrow signal agents stay cheap and deterministic.
4. **Guardrails and HITL as a state-machine guarantee, not a prompt** : hard rules evaluated
   AFTER the agent proposal, fail-closed on internal error, immune to prompt persuasion; a
   durable pending-approval queue in Postgres that physically blocks execution and survives restarts.
5. **Retrieval isolation enforced at the query level** : an unscoped query is not expressible;
   cross-customer contamination is impossible by construction, not by convention.

---

# PART 2 : RESEARCH 

## 2.1 Papers

| # | Title  | Link | What it supports in our architecture |
|---|---|---|---|
| 1 | **MemGPT: Towards LLMs as Operating Systems** (Packer, Wooders, Lin, Fang, Patil, Stoica, Gonzalez) | https://arxiv.org/abs/2310.08560 | Hot/cold memory framing at the state-board level; rejected per-agent LLM self-paging for narrow signal agents |
| 2 | **Ragas: Automated Evaluation of Retrieval Augmented Generation** (Es, James, Espinosa-Anke, Schockaert) | https://arxiv.org/abs/2309.15217 | Stage decomposition of evaluation : retrieval vs generation scored SEPARATELY; never one blended number |
| 3 | **Beyond Autonomy: A Dynamic Tiered AgentRunner Framework for Governable and Resilient Enterprise AI Execution** (Pan, Hou, arXiv May 2026) | https://arxiv.org/abs/2605.10223 | Governance as architecture: risk-adaptive tiering, separation of proposal/review/execution with physically isolated boundaries -> three-tier autonomy + durable blocking pending-approval |
| 4 | **A Unified Framework for Human AI Collaboration in Security Operations Centers with Trusted Autonomy** (Mohsin, Janicke, Ibrahim, Sarker, Camtepe; ACM TOIT 2026) | https://arxiv.org/abs/2505.23397 | Autonomy levels mapped to HITL roles + trust thresholds -> escalation-on-risk-signal; confidence-band calibration |
| 5 | **Cognitive Architectures for Language Agents (CoALA)** (Sumers, Yao, Narasimhan, Griffiths) | https://arxiv.org/abs/2309.02427 | Working/episodic/semantic/procedural taxonomy |
| 6 | **Generative Agents: Interactive Simulacra of Human Behavior** (Park et al., ACM UIST 2023) | https://arxiv.org/abs/2304.03442 | Three-tier memory; adopted the tiering concept, rejected the periodic reflection loop (trigger-driven instead) |
| 7 | **A Bayesian Approach to Graphical Record Linkage and Deduplication** (Steorts, Hall, Fienberg, JASA 2016) | https://www.tandfonline.com/doi/full/10.1080/01621459.2015.1092946 | Graphical record linkage -> transitive closure in the identity graph |
| 8 | **Memory for Autonomous LLM Agents** survey | https://arxiv.org/abs/2603.07670 | Corroborates the CoALA taxonomy application |
| 9 | **Externalization in LLM Agents** survey | https://arxiv.org/abs/2604.08224 | HITL placement patterns |
| 10 | **Awesome-Memory-for-Agents** survey collection (Zep, Mem0, Letta, Graphiti) | https://github.com/NirDiamant/awesome-memory-for-agents | Temporal-KG data model (valid-from / observed-at / superseded-by) inside Postgres without a second stateful system |

## 2.2 Official documentation

| # | Title | Link | What it supports |
|---|---|---|---|
| 1 | **Apache Flink - Timely Stream Processing** (event time, watermarks, lateness) | https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/time/ | Watermark semantics without the engine: bounded out-of-orderness, allowed-lateness re-emission, idle sources |
| 2 | **OpenTelemetry GenAI semantic conventions** | https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai | Vendor-neutral span schema for model/tool/retrieval hops plus custom attributes |
| 3 | **Redpanda Quickstart** (Kafka-API compatible streaming) | https://docs.redpanda.com/current/get-started/quick-start/ | Single-binary event bus for a demonstrable, replayable design |
| 4 | **PostgreSQL full-text search** (ranking, query syntax) | https://www.postgresql.org/docs/current/textsearch-controls.html | Lexical half of hybrid retrieval |

## 2.3 Engineering blogs

| # | Title | Link | What it supports |
|---|---|---|---|
| 1 | **The Live Index: Why Vector Search Should Be a Streaming Problem** (RisingWave) | https://risingwave.com/blog/  | Vector index freshness as a *streaming* problem |
| 2 | **Backfilling in RisingWave: From Historical Initialization to Continuous Streaming** | https://risingwave.com/blog/ | History-seed &  baseline |
| 3 | **From Stored Procedures to Streaming SQL: How Radicant Rebuilt Its Core Banking Transaction Pipeline on RisingWave** | https://risingwave.com/blog/ | Domain evidence: real banks run core transaction pipelines on streaming SQL : validates the event-bus + event-time consumer choice |
| 4 | **Entity Resolution Explained: Top 12 Techniques, Practical Guide & 5 Pythons/R Libraries** | https://spotintelligence.com/2024/01/22/entity-resolution/  | Deterministic-first -> probabilistic -> clustering pipeline -> the identity-resolution order |
| 5 | **How to Find the Producers and Consumers of a Kafka Topic** (Conduktor) | https://www.conduktor.io/blog | Consumer-group semantics, ownership, tracing |
| 6 | **How to Approach Multi-Tenancy in Kafka** (Conduktor) | https://www.conduktor.io/blog | Multi-tenancy patterns -> per-customer isolation of deduplication keys |
| 7 | **Change Data Capture Explained** (Conduktor) | https://www.conduktor.io/blog | CDC |
| 8 | **Airbyte blog** (real-time RAG guidance) | https://airbyte.com/blog | Real-time RAG = change detection |
| 9 | **LangGraph multi-agent concepts: supervisor vs swarm** | https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/ | Supervisor topology (single-trace auditability) vs swarm handoffs |
| 10 | **Wikipedia** (probabilistic linkage, Fellegi–Sunter, blocking) | https://en.wikipedia.org/wiki/Record_linkage | Fellegi–Sunter log-likelihood-ratio scoring with m/u probabilities 

## 2.4 Domain research (existing solutions)

| # | Solution / approach | Link | Takeaway for us |
|---|---|---|---|
| 1 | **Zep / Graphiti** temporal knowledge graphs | https://github.com/getzep/graphiti | Episodic-to-semantic consolidation with point-in-time correctness, adopted the *data model* (valid-from / superseded-by) inside Postgres, deferred the second stateful system |
| 2 | **Mem0** memory layer | https://github.com/mem0ai/mem0 | Memory tiers for agents; reinforced episodic provenance fields |
| 3 | **Letta (MemGPT)** | https://github.com/letta-ai/letta | LLM-driven paging |
| 4 | **Fellegi–Sunter probabilistic linkage** | https://en.wikipedia.org/wiki/Record_linkage | m/u log-likelihood-ratio scoring, probabilistic fallback with stored, auditable confidence |
| 5 | **Apache Flink watermarks** | https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/time/ | The semantics replicated without the engine |

## 2.5 LLM chats

All design refinement and implementation was done through interactive LLM-assisted
sessions with **Claude** (local interactive sessions; the full artifact is this document). What was resolved in them:

| Session (phase) | What was asked | What came out of it |
|---|---|---|
| Phase 0 | Bootstrap the project, seed the research log | Problem framing |
| Phase 1 | Dataset-exact event contracts; replay impairment knobs | Envelope + payload design; SIMULATION-labeled replay; dataset as primary source |
| Phase 2 | State-board concurrency + temporal validity | Compare-and-swap state versioning; episodic supersession; lazy half-life decay; store transaction semantics documented as an implementation discovery |
| Phase 3 | Watermark semantics without a stream-processing engine | Bounded out-of-orderness + idle timeout; arrival-order lateness processing; horizon-based recomputation |
| Phase 4 | Statistical-first signal agents | Z-score detectors against baselines; lexicon sentiment; KYC zero-LLM contract; baseline sampling-bias fix (empty windows treated as valid zero counts) |
| Phase 5 | Hybrid retrieval + structural isolation | Vector + full-text + recency re-ranking; customer scoping enforced structurally at the query level |
| Phase 6 | Trigger economics + life-event inference | Correlation-window triggers + cooldown; structured LLM output with enum-validated retries; disagreement handling |
| Phase 7 | Bounded action selection | Dataset-exact action space; eligibility gating; explicit no-action (zero silence) |
| Phase 8 | Guardrails + durable HITL | Post-proposal hard rules, fail-closed, injection-immune; restart-durable pending approvals |
| Phase 9 | Explainability | Deterministic assembly from decision-log snapshots; zero-LLM by design |
| Phase 10 | Observability | GenAI spans; W3C trace context across the bus; PII scanning; provider set-once semantics documented as an implementation discovery |
| Phase 11 | Evaluation | Fixture-verified metrics (hand-computed calibration/precision/recall) BEFORE real runs; separate no-action precision/recall |

## 2.6 Colab / Kaggle notebooks

| # | Notebook-style resource | Link | Role |
|---|---|---|---|
| 1 | **pgvector examples** (hybrid search with reciprocal rank fusion; retrieval-augmented generation walkthroughs) | https://github.com/pgvector/pgvector/tree/master/examples | The rank-fusion hybrid-search pattern (vector + lexical, then a recency penalty) |
| 2 | **LangGraph tutorials** (supervisor multi-agent) | https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/ | Supervisor topology walkthrough before designing our own typed state graph |

---

# PART 3: IMPLEMENTATION PLAN

## 3.1 The phased plan

| Phase | Delivers |
|---|---|
| 0 | Problem framing, infrastructure plan, research log seeded, architecture decisions recorded |
| 1 | Dataset-exact event contracts; a replay simulator with configurable out-of-order / duplicate / late injection (clearly labeled as simulation); scenario sources |
| 2 | Identity resolution: deterministic-first, probabilistic fallback, identity graph with transitive closure |
| 3 | Customer state board: versioned state, episodic store with temporal validity, lazy decay, conflict log |
| 4 | Streaming semantics: watermark tracking, windowed rolling baselines, allowed-lateness recomputation, deduplication |
| 5 | Signal agents: statistical-first detection against baselines; LLM used only for narrative on flags; KYC fully deterministic |
| 6 | Hybrid retrieval: lexical + vector with recency re-ranking, incremental indexing, structurally-enforced customer scoping |
| 7 | Trigger evaluation (cost-disciplined) + life-event inference with structured output, retries, and disagreement arbitration |
| 8 | Bounded action selection with deterministic eligibility and an explicit no-action |
| 9 | Guardrail state machine (post-proposal, fail-closed) + three-tier autonomy + durable HITL |
| 10 | Deterministic explainability assembled from snapshotted evidence |
| 11 | Observability (GenAI spans, trace propagation across the bus) + PII protection |
| 12 | Evaluation methodology: staged decomposition, fixture-verified metrics first, checkpoint scoring |
| 13 | End-to-end orchestration of the full pipeline + scenario walkthroughs for all scenarios |
| 14 | Documentation written from the design; solution summary; research log finalized |

## 3.2 Pivoted / conditioned from the original plan

- **LLM provider:** Groq instead of Anthropic/OpenAI 
- **Embeddings:** deterministic local hashed-n-gram vectors instead of an external embeddings
  API (none available)
- **Orchestration:** our own typed state-graph design instead of LangGraph (same semantics:
  explicit nodes, persisted checkpoints, replayable traces; LangGraph is the upgrade path)
- **HITL interface:** a minimal operator interface over a durable Postgres queue instead of a
  web service framework
- **Scenario source:** provided dataset primary + synthetic supplements for uncovered
  archetypes (conflicting-signals, ambiguous, fraud-like, no-action-stable)
- **Postgres/pgvector everywhere** instead of a dedicated vector DB, temporal knowledge
  graph, or a Flink cluster

## 3.3 How the design is validated (evaluation methodology)

- **Staged decomposition:** each capability, retrieval, detection, inference, decisioning,
  guardrails, is evaluated separately; never one blended score (the RAGAS principle)
- **Fixture-first metrics:** every metric is hand-computed on small fixtures and verified
  before any real run
- **Checkpoint scoring:** periodic checkpoints are scored against the dataset's ground-truth
  moments (inferred state, confidence band, action, HITL status) using the dataset's fixed enums
- **Adversarial probes:** prompt injection against guardrails, cross-customer leakage against
  retrieval, LLM-failure fallbacks, restart durability of pending approvals, flood resistance
  of the trigger evaluator
- **Timeliness discipline:** watermark lag is observable; late events recompute within
  allowed lateness without double counting
- **Zero silence:** every decision cycle produces an explicit action or an explicit no-action
- **Calibration:** confidence bands are spot-checked against observed accuracy

**What an end-to-end walkthrough would show:** each customer's state is seeded from a 375-event
history, the live stream is replayed through the full pipeline, and checkpoints are emitted
at the ground-truth moments. Every checkpoint carries an explicit action or an explicit
no-action, a confidence band, HITL routing, notes, and resolvable evidence references. All
LLM-dependent steps have a deterministic fallback, so the pipeline runs fully
deterministically without a live LLM.

---
