# Engineering Improvement Plan (Senior Review)

## Objective
Stabilize the current Graph-RAG app and evolve it into a production-ready, observable, and testable platform.

## Why this plan
The repository currently has clear product potential, but there are foundational consistency gaps that should be resolved first:
- The README describes a FastAPI `app/` architecture, while runtime code is Flask in `app.py`.
- The Dockerfile attempts to copy and run `app/main.py` from a non-existent `app/` directory.
- The backend currently interpolates user-derived text into Cypher query strings.

These issues create deployment risk, maintenance friction, and security exposure.

## Prioritized roadmap

### Phase 0 (Day 0-2): Align architecture & make builds deterministic
1. **Choose one backend framework** (FastAPI preferred for async + OpenAPI ergonomics) and remove stale references.
2. **Fix container contract** so Docker build runs what exists in the repository.
3. **Introduce a backend package layout** (`backend/` with modules) to replace monolithic scripts.
4. **Pin runtime versions + startup checks** for Gemini and Neo4j configuration.

**Success criteria**
- `docker build` succeeds.
- `docker run` serves API and frontend in one consistent way.
- README setup instructions exactly match executable commands.

### Phase 1 (Week 1): Security + reliability hardening
1. **Parameterize Cypher queries** (no string interpolation from entity values).
2. **Add input validation and file-size/type limits** for uploads.
3. **Add structured error handling** and typed API contracts for all endpoints.
4. **Add retries/timeouts/circuit-breaker behavior** for Gemini and Neo4j calls.

**Success criteria**
- No dynamic query string interpolation from user-derived values.
- Upload endpoint enforces explicit allowlist and max size.
- 4xx vs 5xx behavior is consistent and documented.

### Phase 2 (Week 2): Testing and CI foundation
1. **Add unit tests** for entity extraction, context retrieval, and response shaping.
2. **Add integration tests** for `/api/upload` and `/api/chat` with mocked LLM/Neo4j.
3. **Add CI pipeline** (lint + tests + build) with required status checks.
4. **Add smoke test** for Docker image startup.

**Success criteria**
- CI runs on every PR.
- Minimum 70% backend coverage on core logic.
- Broken container or API contract fails CI.

### Phase 3 (Week 3): Product quality + observability
1. **Introduce request tracing and metrics** (latency, token usage, retrieval hit-rate).
2. **Add structured logs with correlation IDs** across upload/chat flows.
3. **Implement user-visible job status** for long ingestion tasks.
4. **Capture evaluation baselines** (answer groundedness, context precision).

**Success criteria**
- Dashboard for p50/p95 latency and error rate.
- Every response traceable to query, entities, and retrieval path.
- Ingestion progress visible from frontend.

### Phase 4 (Week 4+): Retrieval quality improvements
1. **Upgrade to hybrid retrieval** (graph traversal + vector/BM25 fallback).
2. **Rerank with cross-encoder** before answer generation.
3. **Implement source citation contract** in model output.
4. **Add offline eval harness** for regression protection.

**Success criteria**
- Measurable quality gain on benchmark question set.
- Citation coverage enforced in answer format.
- Weekly quality report trendline.

## Workstream ownership (recommended)
- **Platform Engineer:** Docker, config, CI/CD, observability.
- **Backend Engineer:** API contracts, graph retrieval, error handling.
- **ML Engineer:** prompt strategy, eval set, reranking.
- **Frontend Engineer:** ingestion status UX, source/citation rendering.

## Risk register
1. **Framework migration churn:** mitigate with adapter layer and parallel endpoints.
2. **Third-party API volatility:** isolate Gemini client behind service interface.
3. **Neo4j availability in dev:** provide local compose profile and mocked mode.
4. **Scope creep:** enforce weekly milestone demo + success criteria gate.

## First 5 concrete tickets to open now
1. `infra: fix Dockerfile to run current backend entrypoint`
2. `docs: reconcile README architecture with actual code`
3. `security: replace Cypher interpolation with parameters`
4. `test: add API integration tests for upload/chat`
5. `observability: add request-id and structured logging middleware`
