# EDMS RAG Jira Ticket Pack

This file converts the current upgrade board and system architecture into Jira-ready tickets.

Assumptions:
- Existing completed work remains out of scope unless a follow-up hardening ticket is needed.
- Tickets focus on the next delivery phase for the Mayan EDMS + RAG sidecar stack.
- Priorities use `Highest`, `High`, `Medium`.

---

## Epic: RAG Search Foundation Hardening

### Ticket 1
**Summary:** Implement persistent Qdrant deployment for RAG index storage

**Issue Type:** Story

**Priority:** Highest

**Labels:** `rag`, `qdrant`, `infra`, `search`

**Description:**
The current search engine uses Qdrant in in-memory mode, which loses embeddings and chunk metadata on sidecar restart. Replace it with a persistent Qdrant server deployment aligned with the target architecture so indexed data survives restarts and can support larger multi-document workloads.

**Acceptance Criteria:**
- Qdrant runs as a dedicated persistent service in Docker Compose.
- RAG sidecar connects to external Qdrant instead of in-memory mode.
- Collections are created automatically on startup with expected vector dimensions and distance metric.
- Restarting the sidecar does not remove indexed documents.
- Restarting the full stack preserves index data through mounted volumes.
- Operational runbook documents ports, persistence path, health checks, and recovery steps.

**Dependencies:** None

---

### Ticket 2
**Summary:** Support append-only multi-document indexing in the RAG pipeline

**Issue Type:** Story

**Priority:** Highest

**Labels:** `rag`, `indexing`, `search`, `mayan`

**Description:**
The current sync flow replaces prior indexed content during document ingestion. Update indexing so documents are appended, updated, and deleted independently, enabling a true shared search corpus across Mayan EDMS documents.

**Acceptance Criteria:**
- Syncing one document does not remove previously indexed documents.
- Re-syncing an existing document updates only that document’s chunks and metadata.
- Each indexed chunk contains a stable document identifier and version-aware metadata.
- Search can return results from multiple documents in a single query.
- A document delete or de-index operation removes only that document’s chunks.
- Automated tests cover first sync, re-sync, and multi-document search behavior.

**Dependencies:** Implement persistent Qdrant deployment for RAG index storage

---

### Ticket 3
**Summary:** Add document-scoped search filters to chat and search APIs

**Issue Type:** Story

**Priority:** High

**Labels:** `rag`, `ui`, `api`, `search`

**Description:**
Users need the option to search within a specific document or across the full corpus. Add document filters through the sidecar API and expose them in the native Mayan chat UI.

**Acceptance Criteria:**
- Search API accepts optional document scope filters.
- Hybrid retrieval applies the filter consistently to vector, BM25, and reranking stages.
- Chat UI includes a selector for `All documents` and individual indexed documents.
- Search results clearly show document identity when cross-document search is used.
- Empty-state and error messaging are handled for filtered queries.
- Tests cover scoped and unscoped retrieval.

**Dependencies:** Support append-only multi-document indexing in the RAG pipeline

---

### Ticket 4
**Summary:** Benchmark and upgrade embedding model for retrieval quality

**Issue Type:** Story

**Priority:** Medium

**Labels:** `rag`, `embeddings`, `evaluation`, `ml`

**Description:**
The current embedding model favors lower memory usage. Run a retrieval benchmark comparing the current model against a stronger alternative and adopt the better option if it improves quality within acceptable infrastructure limits.

**Acceptance Criteria:**
- A representative evaluation dataset is defined from EDMS documents and queries.
- Retrieval quality metrics are captured for the current and candidate embedding models.
- Memory, latency, and indexing throughput are measured for both models.
- A recommendation is documented with clear tradeoffs.
- If the new model is approved, configuration and collection setup are updated accordingly.

**Dependencies:** Implement persistent Qdrant deployment for RAG index storage

---

## Epic: Mayan Automation And UX

### Ticket 5
**Summary:** Trigger automatic RAG indexing from Mayan upload and OCR workflows

**Issue Type:** Story

**Priority:** Highest

**Labels:** `mayan`, `workflow`, `automation`, `rag`

**Description:**
Indexing should happen automatically after document upload and OCR completion rather than relying on manual sync. Configure Mayan workflow actions to invoke the sidecar webhook at the correct processing stage.

**Acceptance Criteria:**
- Mayan workflow sends webhook events after OCR or text extraction completes.
- Webhook payload contains enough identifiers to fetch document content and metadata.
- Duplicate webhook deliveries are handled idempotently.
- Failed indexing attempts are logged and retriable.
- An end-to-end demo proves upload to searchable document with no manual step.
- Documentation explains required Mayan workflow configuration.

**Dependencies:** Support append-only multi-document indexing in the RAG pipeline

---

### Ticket 6
**Summary:** Fix PDF highlight rendering and citation navigation in Mayan chat

**Issue Type:** Bug

**Priority:** High

**Labels:** `mayan`, `ui`, `pdf`, `citations`

**Description:**
Server-side bounding box highlights are generated, but browser rendering and navigation are unreliable. Resolve the client-side issue so source citations consistently open the correct page with visible highlights.

**Acceptance Criteria:**
- Clicking a citation opens the correct document page in the embedded viewer.
- Highlight overlays render correctly for supported documents and browsers.
- Cross-origin and PDF rendering issues are resolved or documented with a fallback.
- The UI gracefully handles missing or malformed bounding boxes.
- Manual verification covers at least Chrome and one additional browser.

**Dependencies:** None

---

### Ticket 7
**Summary:** Implement chat history and user session persistence

**Issue Type:** Story

**Priority:** High

**Labels:** `chat`, `sessions`, `mayan`, `rag`

**Description:**
Persist chat sessions per user so conversations can be resumed, audited, and exported. This aligns the chat feature with the broader EDMS requirement for durable user workflows and stateful sessions.

**Acceptance Criteria:**
- Conversations are stored with user identity, timestamps, and document scope context.
- Users can resume previous chat sessions from the Mayan UI.
- A new chat can be started without overwriting existing history.
- Export of a chat transcript is supported in at least one durable format.
- Retention and deletion rules are documented.
- Basic access controls prevent one user from reading another user’s chats.

**Dependencies:** None

---

### Ticket 8
**Summary:** Improve Mayan RAG chat UX for loading, stats, errors, and mobile layouts

**Issue Type:** Story

**Priority:** Medium

**Labels:** `ui`, `ux`, `mayan`, `rag`

**Description:**
Refine the native Mayan chat app to expose real indexing stats, better user feedback, and a responsive layout that remains usable on smaller screens.

**Acceptance Criteria:**
- UI shows actual indexed document and chunk counts from the sidecar.
- Indexing and search actions display loading states and progress feedback.
- User-facing error messages are specific and actionable.
- Layout remains usable on tablet and mobile widths.
- Visual regressions are checked for the chat panel and PDF viewer states.

**Dependencies:** Support append-only multi-document indexing in the RAG pipeline

---

## Epic: Multimodal And Retrieval Quality

### Ticket 9
**Summary:** Add multimodal image extraction and caption indexing for scanned PDFs

**Issue Type:** Story

**Priority:** High

**Labels:** `rag`, `multimodal`, `ocr`, `vision`

**Description:**
Some documents contain diagrams, stamps, screenshots, or scanned regions with poor text recall. Extend ingestion to extract images, generate captions with a vision-capable model, and index them as searchable chunks.

**Acceptance Criteria:**
- PDF images or image-heavy regions are extracted during ingestion.
- Captions are generated through a configurable vision model provider.
- Caption chunks are stored with source page and image metadata.
- Search can retrieve caption-derived matches alongside text chunks.
- The answer generation path can cite image-derived evidence clearly.
- Provider failures degrade gracefully without blocking text indexing.

**Dependencies:** Support append-only multi-document indexing in the RAG pipeline

---

### Ticket 10
**Summary:** Build a retrieval evaluation suite for RAG relevance and citation accuracy

**Issue Type:** Story

**Priority:** Medium

**Labels:** `rag`, `evaluation`, `quality`, `testing`

**Description:**
The architecture targets industry-grade RAG, but quality is currently judged manually. Create a repeatable evaluation suite that measures retrieval relevance, answer grounding, and citation correctness across representative EDMS documents.

**Acceptance Criteria:**
- Evaluation dataset includes representative questions, expected source pages, and target documents.
- Retrieval metrics include at least Recall@K or MRR for search results.
- Answer evaluation checks groundedness and citation correctness.
- Results can be run repeatedly in local or CI environments.
- Baseline scores are recorded for the current system.

**Dependencies:** Support append-only multi-document indexing in the RAG pipeline

---

## Epic: Security, Governance, And Operations

### Ticket 11
**Summary:** Enforce Mayan ACL-aware filtering in RAG search results

**Issue Type:** Story

**Priority:** Highest

**Labels:** `security`, `acl`, `mayan`, `rag`

**Description:**
The target architecture requires object-level access control. Ensure RAG search and chat only return documents the current authenticated Mayan user is allowed to read.

**Acceptance Criteria:**
- Search requests carry user context from the Mayan app to the sidecar.
- Sidecar validates or consumes document ACL metadata before returning results.
- Unauthorized documents are excluded from retrieval and answer context.
- Access control behavior is covered by tests for allow and deny cases.
- Failure modes default to deny rather than broad exposure.

**Dependencies:** Support append-only multi-document indexing in the RAG pipeline

---

### Ticket 12
**Summary:** Add RAG audit logging for searches, syncs, and chat actions

**Issue Type:** Story

**Priority:** High

**Labels:** `audit`, `observability`, `security`, `rag`

**Description:**
The architecture includes an audit service and analytics path, but current RAG operations are not formally captured. Add audit logging for indexing, searches, answers, and administrative actions.

**Acceptance Criteria:**
- Search, sync, indexing, and chat actions emit structured audit events.
- Events include actor, timestamp, document scope, and outcome status.
- Sensitive content is redacted or minimized in logs.
- Failed operations are logged with diagnostic context.
- Audit storage destination and retention policy are documented.

**Dependencies:** None

---

### Ticket 13
**Summary:** Add API gateway protections for the RAG sidecar endpoints

**Issue Type:** Story

**Priority:** Medium

**Labels:** `security`, `gateway`, `rate-limiting`, `infra`

**Description:**
The architecture assumes a gateway with TLS termination, auth verification, and rate limiting. Harden the RAG sidecar exposure by routing it through a gateway policy instead of broad CORS and direct access assumptions.

**Acceptance Criteria:**
- RAG endpoints are routed through the chosen gateway or protected reverse proxy.
- TLS termination and origin policy are defined for browser-facing requests.
- Rate limits are configured for search, sync, and webhook endpoints.
- Direct unauthenticated access paths are removed or explicitly restricted.
- Deployment documentation includes networking and security assumptions.

**Dependencies:** None

---

### Ticket 14
**Summary:** Add operational health checks, metrics, and failure dashboards for RAG services

**Issue Type:** Story

**Priority:** Medium

**Labels:** `observability`, `infra`, `rag`, `operations`

**Description:**
The system is moving toward a multi-service architecture. Add service health visibility and key operational metrics so failures in indexing, vector storage, and LLM calls are diagnosable.

**Acceptance Criteria:**
- Health endpoints report readiness for sidecar dependencies such as Qdrant and Mayan connectivity.
- Metrics cover indexing throughput, search latency, error rates, and queue backlog if present.
- Logs are structured enough to correlate a request across Mayan and sidecar boundaries.
- Dashboard or documented queries exist for common failures.
- Alerts or alerting hooks are defined for critical outage states.

**Dependencies:** Implement persistent Qdrant deployment for RAG index storage

