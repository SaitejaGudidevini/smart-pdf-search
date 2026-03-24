# EDMS System Architecture - Comprehensive Design Document

## Table of Contents

1. [High-Level Architecture Overview](#1-high-level-architecture-overview)
2. [Core Services & Components](#2-core-services--components)
3. [Data Flow Diagrams](#3-data-flow-diagrams)
4. [Data Models](#4-data-models)
5. [Search Architecture Deep Dive](#5-search-architecture-deep-dive)
6. [Storage Architecture](#6-storage-architecture)
7. [Scalability & Performance](#7-scalability--performance)
8. [Security Architecture](#8-security-architecture)
9. [Deployment Architecture](#9-deployment-architecture)
10. [Technology Stack Recommendations](#10-technology-stack-recommendations)

---

## 1. High-Level Architecture Overview

### 1.1 System Context (C4 Level 1)

```
+-------------------------------------------------------------------+
|                        EXTERNAL ACTORS                            |
|  [End Users]    [Admin Users]    [External APIs]    [SSO/IdP]     |
+-------|--------------|--------------|--------------|--------------+
        |              |              |              |
        v              v              v              v
+===================================================================+
||                     API GATEWAY / LOAD BALANCER                 ||
||  (NGINX/Traefik + Rate Limiting + TLS Termination)              ||
+===================================================================+
        |              |              |              |
        v              v              v              v
+-------------------------------------------------------------------+
|                    APPLICATION LAYER (Services)                   |
|                                                                   |
|  +-------------+ +-------------+ +-------------+ +-------------+ |
|  | Auth        | | Document    | | Search      | | Workflow    | |
|  | Service     | | Service     | | Service     | | Engine      | |
|  +-------------+ +-------------+ +-------------+ +-------------+ |
|  +-------------+ +-------------+ +-------------+ +-------------+ |
|  | OCR/Text    | | RAG/AI      | | Notification| | Audit       | |
|  | Extraction  | | Pipeline    | | Service     | | Service     | |
|  +-------------+ +-------------+ +-------------+ +-------------+ |
+-------------------------------------------------------------------+
        |              |              |              |
        v              v              v              v
+-------------------------------------------------------------------+
|                    MESSAGE BROKER / EVENT BUS                     |
|              (RabbitMQ / Redis Streams / NATS)                    |
+-------------------------------------------------------------------+
        |              |              |              |
        v              v              v              v
+-------------------------------------------------------------------+
|                       DATA LAYER                                  |
|                                                                   |
|  +-------------+ +-------------+ +-------------+ +-------------+ |
|  | PostgreSQL  | | MinIO/S3    | | Qdrant      | | Redis       | |
|  | (Metadata,  | | (File       | | (Vector     | | (Cache,     | |
|  |  ACLs,      | |  Storage)   | |  Embeddings)| |  Sessions,  | |
|  |  Workflows) | |             | |             | |  Queues)    | |
|  +-------------+ +-------------+ +-------------+ +-------------+ |
|  +-------------+ +-------------+                                  |
|  | Meilisearch | | ClickHouse  |                                  |
|  | (Full-Text  | | (Audit Logs |                                  |
|  |  Search)    | |  Analytics) |                                  |
|  +-------------+ +-------------+                                  |
+-------------------------------------------------------------------+
```

### 1.2 Layered Architecture

The system follows a strict 5-layer architecture. Each layer only communicates
with adjacent layers. No layer skipping is permitted.

```
LAYER 1: PRESENTATION
+-------------------------------------------------------------------+
| Web UI (React/Next.js)  |  Mobile App  |  CLI Tool  |  REST API  |
+-------------------------------------------------------------------+
                              |
LAYER 2: API GATEWAY
+-------------------------------------------------------------------+
| TLS Termination | Rate Limiting | Auth Verification | Routing     |
+-------------------------------------------------------------------+
                              |
LAYER 3: APPLICATION SERVICES
+-------------------------------------------------------------------+
| Auth | Documents | Search | OCR | RAG | Workflow | Notify | Audit |
+-------------------------------------------------------------------+
                              |
LAYER 4: DOMAIN / BUSINESS LOGIC
+-------------------------------------------------------------------+
| Document Aggregate | Search Index Model | ACL Engine | Workflow   |
| Version Manager    | Embedding Pipeline | RBAC Model | State Mgr |
+-------------------------------------------------------------------+
                              |
LAYER 5: INFRASTRUCTURE / DATA
+-------------------------------------------------------------------+
| PostgreSQL | S3/MinIO | Qdrant | Redis | Meilisearch | ClickHouse|
+-------------------------------------------------------------------+
```

### 1.3 Back-of-Envelope Estimation for 50,000+ Pages

```
ASSUMPTIONS:
- 50,000 pages across ~5,000 documents (avg 10 pages/doc)
- Average page: 2,500 characters OCR text = ~2.5 KB text
- Average page: 500 KB as scanned image/PDF page
- Average document file: 5 MB (10-page PDF)
- 100 concurrent users, 50 searches/user/day

STORAGE:
- Raw files:        5,000 docs x 5 MB     = 25 GB
- OCR text:         50,000 pages x 2.5 KB  = 125 MB
- Thumbnails:       50,000 pages x 50 KB   = 2.5 GB
- Vector embeddings: 50,000 pages x 6 KB   = 300 MB (1536-dim float32)
- Full-text index:  ~2x OCR text           = 250 MB
- PostgreSQL meta:  ~500 MB
- TOTAL:            ~29 GB (easily fits single server)

TRAFFIC:
- Search QPS:       100 users x 50/day / 86,400 = ~0.06 QPS avg
                    Peak: ~1-2 QPS (bursty)
- Upload:           ~50 docs/day = negligible QPS
- OCR processing:   50 docs x 10 pages = 500 pages/day
                    At 2s/page = ~17 minutes processing/day

SCALING THRESHOLD:
- At 500,000 pages: storage ~290 GB, may need distributed object store
- At 5,000,000 pages: storage ~2.9 TB, need sharding + read replicas
- At 50,000,000 pages: storage ~29 TB, full distributed architecture
```

---

## 2. Core Services & Components

### 2.1 API Gateway / Load Balancer

```
PURPOSE: Single entry point for all client requests.
         Handles cross-cutting concerns before requests reach services.

RESPONSIBILITIES:
  - TLS termination (HTTPS -> HTTP internally)
  - Rate limiting (per-user, per-IP, per-endpoint)
  - Request routing to backend services
  - JWT validation (verify signature, check expiry)
  - CORS handling
  - Request/response logging
  - Compression (gzip/brotli)
  - Health check endpoints

TECHNOLOGY: Traefik (recommended) or NGINX

INPUT:  HTTPS requests from clients
OUTPUT: Routed HTTP requests to internal services

CONNECTIONS:
  Client --> [API Gateway] --> Auth Service (token validation)
                           --> Document Service (CRUD)
                           --> Search Service (queries)
                           --> Workflow Service (actions)
```

**Configuration Example (Traefik):**
```
Routing Rules:
  /api/v1/auth/*        --> auth-service:3001
  /api/v1/documents/*   --> document-service:3002
  /api/v1/search/*      --> search-service:3003
  /api/v1/workflows/*   --> workflow-service:3004
  /api/v1/admin/*       --> admin-service:3005

Rate Limits:
  Global:               1000 req/min per IP
  /api/v1/search:       100 req/min per user
  /api/v1/documents (POST): 20 req/min per user
```

---

### 2.2 Auth Service (RBAC + ACL)

```
PURPOSE: Authentication, authorization, and access control for all
         system operations. Dual-model: role-based (RBAC) for system
         permissions + object-level ACL for document access.

RESPONSIBILITIES:
  - User registration and login (email/password, OAuth2, SAML)
  - JWT issuance (access + refresh tokens)
  - Multi-factor authentication (TOTP)
  - Role management (admin, editor, viewer, custom)
  - Object-level ACL (per-document, per-cabinet permissions)
  - API key management for integrations
  - Session management and token revocation

TECHNOLOGY: Node.js/TypeScript + Passport.js + bcrypt + speakeasy (TOTP)

INPUT:
  - Login credentials (email/password, OAuth2 code, SAML assertion)
  - Authorization check requests (user X, resource Y, action Z)

OUTPUT:
  - JWT tokens (access: 15min, refresh: 7 days)
  - Authorization decisions (allow/deny with reason)
  - User profile data

CONNECTIONS:
  API Gateway --> [Auth Service] --> PostgreSQL (users, roles, ACLs)
                                 --> Redis (sessions, token blocklist)
                                 --> External IdP (OAuth2, SAML)
```

**RBAC Model:**
```
ROLES AND PERMISSIONS MATRIX:

Permission              | Admin | Manager | Editor | Viewer | Uploader
------------------------|-------|---------|--------|--------|--------
document.create         |   Y   |    Y    |   Y    |   N    |   Y
document.read           |   Y   |    Y    |   Y    |   Y    |   N
document.update         |   Y   |    Y    |   Y    |   N    |   N
document.delete         |   Y   |    Y    |   N    |   N    |   N
document.share          |   Y   |    Y    |   Y    |   N    |   N
cabinet.manage          |   Y   |    Y    |   N    |   N    |   N
workflow.create         |   Y   |    Y    |   N    |   N    |   N
workflow.approve        |   Y   |    Y    |   Y    |   N    |   N
user.manage             |   Y   |    N    |   N    |   N    |   N
system.configure        |   Y   |    N    |   N    |   N    |   N
```

**ACL Model (Object-Level):**
```
Each document/cabinet has an Access Control List:

ACL Entry = {
  subject:    user_id | role_id | group_id | "everyone"
  resource:   document_id | cabinet_id
  permission: "read" | "write" | "delete" | "share" | "full_control"
  inherited:  boolean  (from parent cabinet)
  granted_by: user_id
  expires_at: timestamp | null
}

Resolution order (first match wins):
  1. Explicit DENY on resource
  2. Explicit ALLOW on resource
  3. Inherited permissions from parent cabinet
  4. Role-based default permissions
  5. Default: DENY
```

---

### 2.3 Document Ingestion Pipeline

```
PURPOSE: Accepts uploaded files, validates them, stores originals,
         and triggers the async processing pipeline (OCR, indexing,
         thumbnail generation, embedding).

RESPONSIBILITIES:
  - File upload handling (multipart, chunked for large files)
  - File type validation and virus scanning
  - Original file storage to S3/MinIO
  - Metadata extraction (EXIF, PDF properties, Office metadata)
  - Thumbnail generation
  - Version management (new version of existing doc)
  - Triggering async pipeline via message queue

TECHNOLOGY: Node.js/TypeScript + Multer + Sharp (thumbnails) + BullMQ

INPUT:
  - Multipart file upload (PDF, DOCX, XLSX, images, etc.)
  - Metadata (title, description, tags, cabinet assignment)
  - User context (who uploaded, permissions)

OUTPUT:
  - Document record in PostgreSQL (status: "processing")
  - Original file in S3/MinIO
  - Thumbnail images (multiple sizes)
  - Job messages on processing queue

CONNECTIONS:
  API Gateway --> [Document Service] --> PostgreSQL (document metadata)
                                     --> S3/MinIO (file storage)
                                     --> Message Queue (processing jobs)
                                     --> Auth Service (ACL creation)
```

**Upload Processing Sequence:**
```
1. Client sends multipart POST /api/v1/documents
2. API Gateway validates JWT, forwards to Document Service
3. Document Service:
   a. Validates file type against allowlist
   b. Scans for viruses (ClamAV integration)
   c. Generates document UUID
   d. Stores original to S3: /originals/{doc_uuid}/{version}/file.ext
   e. Extracts basic metadata (file size, MIME type, page count)
   f. Creates Document + DocumentVersion records in PostgreSQL
   g. Creates default ACL (owner = uploader, full_control)
   h. Generates thumbnails (150px, 300px, 600px per page)
   i. Stores thumbnails to S3: /thumbnails/{doc_uuid}/{page_num}/
   j. Publishes job to "document.processing" queue:
      {
        document_id: uuid,
        version_id: uuid,
        file_path: "s3://originals/...",
        file_type: "application/pdf",
        page_count: 10,
        tasks: ["ocr", "index", "embed"]
      }
   k. Returns 202 Accepted with document_id and status "processing"
```

---

### 2.4 Storage Service

```
PURPOSE: Abstraction layer over file and object storage. Provides
         consistent API for storing, retrieving, and managing binary
         files regardless of underlying storage backend.

RESPONSIBILITIES:
  - Store/retrieve binary files (originals, thumbnails, exports)
  - Presigned URL generation for direct client downloads
  - Storage quota management per user/organization
  - File deduplication via content-addressable hashing (SHA-256)
  - Lifecycle management (move old versions to cold storage)

TECHNOLOGY: MinIO (S3-compatible, self-hosted) or AWS S3

INPUT:
  - File buffer/stream + storage path
  - Retrieval requests (by path, with presigned URL option)

OUTPUT:
  - Storage confirmation (path, size, hash)
  - Presigned download URLs (15-minute expiry)
  - Storage usage metrics

BUCKET STRUCTURE:
  edms-originals/
    /{document_uuid}/{version_num}/original.{ext}
  edms-thumbnails/
    /{document_uuid}/{version_num}/{page_num}_{size}.jpg
  edms-ocr-text/
    /{document_uuid}/{version_num}/full.txt
    /{document_uuid}/{version_num}/pages/{page_num}.txt
  edms-exports/
    /{export_uuid}/export.{ext}
  edms-temp/
    /{upload_uuid}/chunk_{n}

CONNECTIONS:
  Document Service --> [Storage Service] --> MinIO/S3 (object storage)
  OCR Pipeline     --> [Storage Service] --> MinIO/S3
  Search Service   --> [Storage Service] --> MinIO/S3 (for presigned URLs)
```

---

### 2.5 OCR & Text Extraction Pipeline

```
PURPOSE: Extracts machine-readable text from uploaded documents.
         Handles PDFs (both native text and scanned images), Office
         documents, and raw images. Produces page-level and line-level
         text with positional metadata (bounding boxes).

RESPONSIBILITIES:
  - PDF text extraction (native text layer)
  - OCR for scanned documents and images
  - Page-by-page text extraction with coordinates
  - Line-level text segmentation with bounding boxes
  - Language detection
  - Table detection and extraction
  - Handwriting recognition (where supported)

TECHNOLOGY:
  - Tesseract 5.x (OCR engine, 100+ languages)
  - pdf-parse / pdf.js (native PDF text extraction)
  - Poppler (PDF to image conversion for OCR)
  - LibreOffice headless (DOCX/XLSX to PDF conversion)
  - Python worker (Tesseract + pytesseract + pdf2image)

INPUT:
  - Job from message queue:
    { document_id, version_id, file_path, file_type, page_count }

OUTPUT:
  - Per-page text content with metadata stored in PostgreSQL
  - Per-line text with bounding box coordinates
  - Full document text file stored in S3
  - Completion event on message queue

CONNECTIONS:
  Message Queue --> [OCR Pipeline] --> S3/MinIO (read file, write text)
                                   --> PostgreSQL (page/line records)
                                   --> Message Queue (completion event)
                                   --> Indexing Service (trigger indexing)
```

**OCR Processing Detail:**
```
For each page in the document:

1. CONVERSION STEP:
   PDF page --> 300 DPI PNG image (via Poppler/pdf2image)
   DOCX    --> PDF (via LibreOffice) --> PNG
   Image   --> use directly

2. TEXT EXTRACTION (dual-path):
   Path A: Native text extraction (for PDFs with text layer)
     - Use pdf-parse to extract text per page
     - If text found and length > threshold: use native text

   Path B: OCR (for scanned docs or images)
     - Run Tesseract with HOCR output (gives bounding boxes)
     - Parse HOCR XML to extract:
       - Page text (full text of the page)
       - Lines: [{text, bbox: {x, y, w, h}, confidence}]
       - Words: [{text, bbox: {x, y, w, h}, confidence}]

3. LINE SEGMENTATION:
   For each page, text is broken into lines:

   Page 1, Line 1:  "INVOICE NUMBER: INV-2024-0042"     bbox(50,100,400,120)
   Page 1, Line 2:  "Date: January 15, 2024"            bbox(50,130,300,150)
   Page 1, Line 3:  "Bill To: Acme Corporation"          bbox(50,160,350,180)
   ...

4. STORAGE:
   - Page record in PostgreSQL:
     { page_id, document_id, version_id, page_number,
       full_text, word_count, language, ocr_confidence }

   - Line records in PostgreSQL:
     { line_id, page_id, line_number, text,
       bbox_x, bbox_y, bbox_w, bbox_h, confidence }

   - Full text file to S3: /ocr-text/{doc_uuid}/{version}/pages/1.txt

5. EMIT EVENT:
   Publish to queue: "document.ocr.complete"
   { document_id, version_id, page_count, total_words, avg_confidence }
```

---

### 2.6 Indexing Service (Page-Level, Line-Level)

```
PURPOSE: Builds and maintains full-text search indexes and vector
         embedding indexes. Creates granular, addressable search
         units at page and line level so search results can point
         users to the exact location within a document.

RESPONSIBILITIES:
  - Full-text indexing (Meilisearch) at page and line granularity
  - Vector embedding generation (OpenAI/local model) per page chunk
  - Index lifecycle management (create, update, delete)
  - Re-indexing on document update/new version
  - Index optimization and compaction

TECHNOLOGY:
  - Meilisearch (full-text search, typo-tolerant, fast)
  - Qdrant (vector database for embeddings)
  - OpenAI text-embedding-3-small or local Sentence-Transformers

INPUT:
  - OCR completion events from message queue
  - Page and line text from PostgreSQL

OUTPUT:
  - Full-text index entries in Meilisearch
  - Vector embeddings in Qdrant
  - Index status updates in PostgreSQL

CONNECTIONS:
  Message Queue  --> [Indexing Service] --> PostgreSQL (read page/line data)
                                        --> Meilisearch (write FT index)
                                        --> Qdrant (write vectors)
                                        --> Embedding API (generate vectors)
```

**Indexing Data Structure:**
```
MEILISEARCH INDEX: "pages"
{
  id:            "page_{page_id}",
  document_id:   "uuid",
  document_title: "Invoice Q4 2024",
  version_id:    "uuid",
  page_number:   3,
  text:          "Full text of page 3...",
  cabinet:       "Finance",
  tags:          ["invoice", "Q4", "2024"],
  uploaded_by:   "user_uuid",
  uploaded_at:   "2024-01-15T10:30:00Z",
  acl_read:      ["user_1", "role_editor", "group_finance"]
}

MEILISEARCH INDEX: "lines"
{
  id:            "line_{line_id}",
  document_id:   "uuid",
  page_id:       "uuid",
  page_number:   3,
  line_number:   7,
  text:          "Total Amount Due: $15,420.00",
  document_title: "Invoice Q4 2024",
  acl_read:      ["user_1", "role_editor", "group_finance"]
}

QDRANT COLLECTION: "page_embeddings"
{
  id:        "page_{page_id}",
  vector:    [0.0123, -0.0456, ...],   // 1536 dimensions
  payload: {
    document_id:    "uuid",
    page_number:    3,
    document_title: "Invoice Q4 2024",
    text_preview:   "First 200 chars of page...",
    cabinet:        "Finance",
    acl_read:       ["user_1", "role_editor"]
  }
}
```

**Chunking Strategy for Embeddings:**
```
STRATEGY: Page-level chunks with overlap

For each page:
  1. If page text < 512 tokens: embed as single chunk
  2. If page text 512-1024 tokens: embed as single chunk
  3. If page text > 1024 tokens: split into overlapping chunks
     - Chunk size: 512 tokens
     - Overlap: 64 tokens
     - Each chunk tagged with page_number + chunk_index

Why page-level (not paragraph or sentence):
  - Pages are natural document boundaries
  - Users think in page numbers ("it was on page 3")
  - Maintains context within a single page
  - Manageable number of vectors (50K pages = 50K-100K vectors)
```

---

### 2.7 Search Service (Keyword + Semantic + Vector)

```
PURPOSE: Unified search interface that combines keyword search,
         semantic (vector) search, and hybrid search with re-ranking.
         Returns results at document, page, or line granularity with
         exact location citations.

RESPONSIBILITIES:
  - Keyword search (full-text with typo tolerance)
  - Semantic search (vector similarity via embeddings)
  - Hybrid search (combine keyword + semantic with RRF)
  - Faceted search (filter by cabinet, tag, date, type)
  - ACL-filtered results (user only sees what they can access)
  - Search result highlighting and snippets
  - Search suggestions and autocomplete

TECHNOLOGY:
  - Meilisearch (keyword search)
  - Qdrant (vector search)
  - Custom ranking/fusion in Node.js

INPUT:
  - Search query (text string)
  - Search mode: "keyword" | "semantic" | "hybrid" | "smart"
  - Filters: { cabinets, tags, date_range, file_types, uploaded_by }
  - Granularity: "document" | "page" | "line"
  - Pagination: { offset, limit }
  - User context (for ACL filtering)

OUTPUT:
  - Ranked search results with:
    - Document metadata (title, cabinet, tags)
    - Page number and line number (for page/line granularity)
    - Highlighted text snippet with match context
    - Relevance score
    - Bounding box coordinates (for visual highlight in viewer)

CONNECTIONS:
  API Gateway --> [Search Service] --> Meilisearch (keyword search)
                                   --> Qdrant (vector search)
                                   --> PostgreSQL (ACL check, metadata)
                                   --> Redis (search cache, suggestions)
```

**Search Modes Explained:**
```
MODE 1: KEYWORD SEARCH
  User types: "invoice amount $15,420"
  Process:
    1. Query Meilisearch "lines" index with text query
    2. Meilisearch handles typo tolerance, tokenization, ranking
    3. Filter results by user's ACL (acl_read field)
    4. Return matching lines with page/document context

MODE 2: SEMANTIC SEARCH
  User types: "documents about quarterly financial obligations"
  Process:
    1. Generate embedding vector for the query string
    2. Query Qdrant for nearest neighbors (cosine similarity)
    3. Filter by ACL in Qdrant payload filter
    4. Return matching pages with similarity scores

MODE 3: HYBRID SEARCH (recommended default)
  User types: "what is the total amount on the Acme invoice?"
  Process:
    1. Run keyword search in Meilisearch (get top 50)
    2. Run semantic search in Qdrant (get top 50)
    3. Fuse results using Reciprocal Rank Fusion (RRF):
       RRF_score(d) = SUM( 1 / (k + rank_i(d)) ) for each system i
       where k = 60 (constant)
    4. Re-rank fused results
    5. Filter by ACL
    6. Return top N results with both match types indicated

MODE 4: SMART SEARCH (page+line precision)
  User types: "find the exact line mentioning payment terms"
  Process:
    1. Run hybrid search at LINE granularity
    2. For each matching line, include:
       - Document title and ID
       - Page number
       - Line number
       - Exact text of the line
       - Bounding box for highlighting in document viewer
       - 2 lines of context above and below
    3. Group results by document, ordered by relevance
```

**Search Response Schema:**
```json
{
  "query": "payment terms net 30",
  "mode": "smart",
  "total_results": 23,
  "results": [
    {
      "document_id": "uuid-1",
      "document_title": "Acme Corp Service Agreement",
      "version": 3,
      "matches": [
        {
          "page_number": 4,
          "line_number": 17,
          "text": "Payment terms: Net 30 days from invoice date",
          "context_before": ["...", "The Client agrees to the following:"],
          "context_after": ["Late payments incur 1.5% monthly interest", "..."],
          "bbox": { "x": 50, "y": 340, "w": 450, "h": 18 },
          "score": 0.94,
          "match_type": "hybrid",
          "highlights": [
            { "offset": 0, "length": 13, "field": "payment terms" },
            { "offset": 15, "length": 6, "field": "Net 30" }
          ]
        }
      ],
      "thumbnail_url": "/api/v1/documents/uuid-1/pages/4/thumbnail",
      "download_url": "/api/v1/documents/uuid-1/download"
    }
  ],
  "facets": {
    "cabinets": [{"name": "Legal", "count": 12}, {"name": "Finance", "count": 11}],
    "file_types": [{"name": "pdf", "count": 18}, {"name": "docx", "count": 5}],
    "years": [{"name": "2024", "count": 15}, {"name": "2023", "count": 8}]
  }
}
```

---

### 2.8 Source-First Smart Query (Replaces Traditional RAG)

```
DESIGN PHILOSOPHY:
  Traditional RAG: LLM generates an answer → user must trust AI → no legitimacy
  Source-First:    System finds exact source → shows ORIGINAL document with
                   highlighted section → floating panel explains context
                   → user sees proof first, explanation second

  PRINCIPLE: "Show the evidence, not the interpretation"
  The document IS the answer. The AI only helps you FIND it.

PURPOSE: Accepts natural language questions, locates the exact document
         sections that contain the answer, and presents the ORIGINAL
         document with highlighted regions alongside a floating context
         panel that summarizes what was found — without generating or
         paraphrasing content.

RESPONSIBILITIES:
  - Accept natural language questions
  - Retrieve exact document locations via hybrid search
  - Return ORIGINAL text with precise coordinates (doc, page, line, bbox)
  - Provide floating context panel with:
    - Which documents matched and why
    - Key extracted facts (not AI-generated — extracted verbatim)
    - Navigation to jump between matches
  - Optional: AI summary clearly labeled as "AI interpretation" (secondary)
  - Enforce ACL (only retrieve docs user can access)

TECHNOLOGY:
  - Qdrant (vector retrieval for semantic matching)
  - Meilisearch (keyword matching for exact terms)
  - Embedding API (query embedding)
  - Redis (result caching)
  - NO LLM required for core functionality (LLM is optional enhancement)

INPUT:
  - Natural language question or keyword query
  - Optional scope filters (cabinets, tags, date range)
  - User context (for ACL)

OUTPUT:
  - Primary: Document viewer showing original document at exact location
  - Floating Panel: Context card with matched sources and extracted facts
  - NO AI-generated text in the primary response

CONNECTIONS:
  API Gateway --> [Smart Query]  --> Qdrant (semantic search)
                                 --> Meilisearch (keyword search)
                                 --> Embedding API (embed question)
                                 --> PostgreSQL (ACL, metadata, line text)
                                 --> Redis (result cache)
                                 --> Storage Service (presigned URLs for viewer)
```

**Source-First Query Pipeline (Step-by-Step):**
```
STEP 1: QUERY UNDERSTANDING (no LLM needed)
  Input: "What are the payment terms in our agreement with Acme?"

  a. Tokenize and extract key entities using rules + NER:
     - entities: ["payment terms", "Acme"]
     - intent: factual lookup

  b. Generate search variants:
     - keyword queries: "payment terms Acme", "payment terms", "net days"
     - semantic query: embed full question as vector

STEP 2: MULTI-LEVEL RETRIEVAL
  Run in parallel:

  a. LINE-LEVEL keyword search (Meilisearch "lines" index):
     Query: "payment terms" + filter: acl_read contains user
     → Returns: exact lines with page/line numbers

  b. PAGE-LEVEL semantic search (Qdrant):
     Query: question embedding vector + ACL filter
     → Returns: pages that are semantically about payment terms

  c. PAGE-LEVEL keyword search (Meilisearch "pages" index):
     Query: "payment terms Acme"
     → Returns: pages containing these keywords

STEP 3: RESULT FUSION & RANKING
  a. Combine all results using Reciprocal Rank Fusion (RRF)
  b. Deduplicate: if line match and page match are same location, merge
  c. Group by document → sort by relevance within each document
  d. For each match, fetch surrounding context:
     - 3 lines above and below the matching line
     - Full page text for page-level matches

STEP 4: BUILD SOURCE EVIDENCE PACKAGE
  For each matched location, construct a "source evidence" object:

  {
    "sources": [
      {
        "rank": 1,
        "relevance_score": 0.94,
        "match_type": "hybrid",          // keyword + semantic both matched

        // EXACT LOCATION
        "document_id": "uuid-1",
        "document_title": "Acme Corp Service Agreement",
        "version": 3,
        "page_number": 4,
        "line_start": 15,
        "line_end": 20,

        // ORIGINAL TEXT (verbatim from document, NOT AI-generated)
        "matched_text": "Payment terms: Net 30 days from invoice date. Late payments incur 1.5% monthly interest. All payments in USD.",
        "context_before": [
          "13: The Client agrees to the following terms:",
          "14: "
        ],
        "context_after": [
          "21: ",
          "22: 5. Intellectual Property"
        ],

        // VISUAL COORDINATES (for highlighting in document viewer)
        "highlights": [
          { "page": 4, "line": 15, "bbox": {"x":50,"y":300,"w":450,"h":18} },
          { "page": 4, "line": 16, "bbox": {"x":50,"y":322,"w":420,"h":18} },
          { "page": 4, "line": 17, "bbox": {"x":50,"y":344,"w":380,"h":18} }
        ],

        // VIEWER URL (opens document at exact page with highlights)
        "viewer_url": "/viewer/uuid-1?page=4&highlight=15-20",
        "thumbnail_url": "/api/v1/documents/uuid-1/pages/4/thumbnail"
      },
      {
        "rank": 2,
        "relevance_score": 0.87,
        "document_title": "Acme Corp Amendment 2024",
        "page_number": 2,
        "line_start": 8,
        "line_end": 12,
        "matched_text": "Amended payment terms: Net 45 days effective January 2024. Early payment discount of 2% if paid within 10 days.",
        "viewer_url": "/viewer/uuid-2?page=2&highlight=8-12",
        ...
      }
    ],

    // EXTRACTED FACTS (verbatim key-value pairs pulled from matched text)
    // These are NOT AI-generated — they are pattern-extracted from source
    "extracted_facts": [
      { "label": "Payment Terms", "value": "Net 30 days (amended to Net 45 days Jan 2024)", "source_rank": [1, 2] },
      { "label": "Late Fee", "value": "1.5% monthly interest", "source_rank": [1] },
      { "label": "Early Payment Discount", "value": "2% if paid within 10 days", "source_rank": [2] },
      { "label": "Currency", "value": "USD", "source_rank": [1] }
    ],

    // METADATA
    "total_sources_found": 3,
    "documents_searched": 1247,
    "search_time_ms": 145,

    // OPTIONAL: AI summary (clearly labeled, secondary, collapsible)
    "ai_summary": {
      "enabled": true,
      "label": "AI Interpretation (verify against sources above)",
      "text": "The payment terms were originally Net 30 but amended to Net 45 in 2024...",
      "disclaimer": "This is an AI-generated summary. Always verify against the original documents shown above."
    }
  }
```

**UI Layout: Floating Context Panel + Document Viewer:**
```
+------------------------------------------------------------------+
|  Search Bar: "What are the payment terms with Acme?"    [Search] |
+------------------------------------------------------------------+
|                          |                                        |
|   FLOATING CONTEXT       |       DOCUMENT VIEWER                  |
|   PANEL (left/right)     |       (main area)                     |
|                          |                                        |
|   ┌──────────────────┐   |   ┌──────────────────────────────────┐ |
|   │ 3 sources found  │   |   │                                  │ |
|   │ in 145ms         │   |   │  Acme Corp Service Agreement     │ |
|   │                  │   |   │  Page 4 of 12                    │ |
|   │ ── Source 1 ──── │   |   │                                  │ |
|   │ 📄 Acme Corp     │   |   │  ...                             │ |
|   │ Service Agreement│   |   │  The Client agrees to the        │ |
|   │ Page 4, Lines    │   |   │  following terms:                │ |
|   │ 15-20            │   |   │                                  │ |
|   │ Score: 94%       │   |   │ ┌──────────────────────────────┐ │ |
|   │ [Jump to ➜]      │   |   │ │ Payment terms: Net 30 days  │ │ |
|   │                  │   |   │ │ from invoice date. Late      │ │ |
|   │ ── Source 2 ──── │   |   │ │ payments incur 1.5% monthly  │ │ |
|   │ 📄 Acme Corp     │   |   │ │ interest. All payments in    │ │ |
|   │ Amendment 2024   │   |   │ │ USD.                         │ │ |
|   │ Page 2, Lines    │   |   │ └──────────────────────────────┘ │ |
|   │ 8-12             │   |   │  (highlighted in yellow)         │ |
|   │ Score: 87%       │   |   │                                  │ |
|   │ [Jump to ➜]      │   |   │  5. Intellectual Property        │ |
|   │                  │   |   │  ...                             │ |
|   │ ── Key Facts ─── │   |   │                                  │ |
|   │ Terms: Net 45d   │   |   └──────────────────────────────────┘ |
|   │ Late: 1.5%/mo    │   |                                        |
|   │ Discount: 2%/10d │   |   [◀ Prev Source]  [Next Source ▶]     |
|   │                  │   |                                        |
|   │ ── AI Summary ── │   |                                        |
|   │ ▼ (collapsed)    │   |                                        |
|   │ "⚠ AI interpret- │   |                                        |
|   │  ation - verify  │   |                                        |
|   │  against sources"│   |                                        |
|   └──────────────────┘   |                                        |
|                          |                                        |
+------------------------------------------------------------------+

INTERACTIONS:
  - Click "Jump to ➜" on any source → viewer scrolls to that page+line
  - Click [Next Source ▶] → viewer jumps to next match
  - Highlighted regions have yellow background with subtle border
  - Key Facts are extracted verbatim, not AI-generated
  - AI Summary is collapsed by default with ⚠ warning label
  - User can toggle AI Summary on/off in settings
```

**Fact Extraction (Rule-Based, No LLM):**
```
PURPOSE: Extract key-value facts from matched text using patterns,
         NOT LLM generation. This ensures 100% legitimacy.

EXTRACTION PATTERNS:
  Financial:
    /(?:net|payment)\s*(?:terms?:?\s*)?(\d+)\s*days?/i
    → { label: "Payment Terms", value: "Net {N} days" }

    /(\d+(?:\.\d+)?)\s*%\s*(?:monthly|annual|per)\s*(interest|fee)/i
    → { label: "Interest/Fee", value: "{N}% {period}" }

    /\$[\d,]+(?:\.\d{2})?/
    → { label: "Amount", value: "${amount}" }

  Dates:
    /effective\s+(\w+\s+\d{1,2},?\s+\d{4})/i
    → { label: "Effective Date", value: "{date}" }

  Entity:
    NER extraction for organization names, people, locations

  Custom:
    Users can define custom extraction patterns per document type
    e.g., Invoice type: extract "Invoice Number", "Due Date", "Total"
```

**Why Source-First > Traditional RAG for EDMS:**
```
TRUST & LEGITIMACY:
  ✗ Traditional RAG: "The AI says payment terms are Net 30"
    → User thinks: "Is this right? What if the AI hallucinated?"
    → User has to go find the original document anyway

  ✓ Source-First: Shows the ACTUAL document page with highlighted text
    → User sees: "Here is the exact clause on page 4, line 15"
    → User trusts it because they're reading the original

LEGAL/COMPLIANCE:
  ✗ RAG answer: Not admissible, not authoritative
  ✓ Original document with citation: Auditable, traceable, admissible

USER WORKFLOW:
  ✗ RAG: Read AI answer → doubt it → search for original → verify
  ✓ Source-First: See original immediately → understand context → done

COST:
  ✗ RAG: Requires LLM API call for every query ($$$)
  ✓ Source-First: Pure retrieval, no LLM needed (LLM optional extra)

ACCURACY:
  ✗ RAG: Can hallucinate, paraphrase incorrectly, miss nuance
  ✓ Source-First: 100% accurate — it's the original text
```

---

### 2.9 Workflow Engine

```
PURPOSE: Manages document lifecycle workflows such as review/approval
         processes, document routing, and automated actions triggered
         by state transitions.

RESPONSIBILITIES:
  - Define workflow templates (states, transitions, actions)
  - Assign documents to workflows
  - Track workflow state per document
  - Execute transition actions (notify, set metadata, move cabinet)
  - Enforce transition rules (who can approve, conditions)
  - Deadline management and escalation
  - Parallel and sequential approval paths

TECHNOLOGY: Node.js/TypeScript + custom state machine + BullMQ (timers)

INPUT:
  - Workflow template definitions (JSON/YAML)
  - Workflow actions (submit, approve, reject, escalate)
  - Document context and user context

OUTPUT:
  - Workflow state changes
  - Notifications to participants
  - Audit log entries
  - Automated actions (metadata update, cabinet move, etc.)

CONNECTIONS:
  API Gateway    --> [Workflow Engine] --> PostgreSQL (workflow state)
  Document Svc   --> [Workflow Engine] --> Notification Service
  Message Queue  --> [Workflow Engine] --> Audit Service
```

**Workflow Definition Example:**
```
WORKFLOW: "Document Review and Approval"

STATES:
  [Draft] --> [Submitted] --> [Under Review] --> [Approved]
                                             --> [Rejected] --> [Draft]
                          --> [Needs Changes] --> [Draft]

TRANSITIONS:
  Draft -> Submitted:
    who: document_owner
    action: notify_reviewers

  Submitted -> Under Review:
    who: assigned_reviewer
    action: lock_document, start_deadline(5_days)

  Under Review -> Approved:
    who: assigned_reviewer (min 2 approvals for sensitive docs)
    action: unlock_document, move_to_cabinet("Approved"),
            set_metadata(approved_date, approved_by),
            notify_owner("Your document has been approved")

  Under Review -> Rejected:
    who: assigned_reviewer
    requires: rejection_comment
    action: unlock_document, notify_owner_with_comments

  Under Review -> Needs Changes:
    who: assigned_reviewer
    requires: change_comments
    action: unlock_document, notify_owner_with_comments

ESCALATION:
  If Under Review > 5 days without action:
    notify_reviewer_manager
  If Under Review > 10 days:
    auto_assign_to_manager
```

---

### 2.10 Notification Service

```
PURPOSE: Delivers notifications to users across multiple channels
         based on system events and user preferences.

RESPONSIBILITIES:
  - Event-driven notifications (workflow state changes, shares, etc.)
  - Multi-channel delivery (in-app, email, webhook)
  - User notification preferences
  - Notification history and read tracking
  - Rate limiting and batching (digest mode)

TECHNOLOGY: Node.js/TypeScript + BullMQ + Nodemailer + WebSocket

INPUT:
  - Notification events from message queue
  - { type, recipient_id, subject, body, channel, metadata }

OUTPUT:
  - In-app notifications (via WebSocket)
  - Email notifications (via SMTP)
  - Webhook calls (for integrations)

CONNECTIONS:
  Message Queue    --> [Notification Service] --> WebSocket (in-app)
                                              --> SMTP Server (email)
                                              --> External Webhooks
                                              --> PostgreSQL (history)
                                              --> Redis (user preferences)
```

---

### 2.11 Audit/Event Log Service

```
PURPOSE: Immutable record of every significant action in the system.
         Provides compliance trail, forensics capability, and usage
         analytics.

RESPONSIBILITIES:
  - Capture all CRUD operations on documents
  - Capture all authentication events (login, logout, failed attempts)
  - Capture all permission changes
  - Capture all workflow state transitions
  - Capture all search queries (for analytics)
  - Immutable append-only storage
  - Query interface for audit reports
  - Retention policy enforcement

TECHNOLOGY: ClickHouse (columnar, fast analytics) + Node.js

INPUT:
  - Audit events from message queue:
    {
      event_id:    uuid,
      timestamp:   ISO-8601,
      actor_id:    user_uuid,
      actor_ip:    "192.168.1.100",
      action:      "document.download",
      resource_type: "document",
      resource_id: document_uuid,
      details:     { version: 3, format: "pdf" },
      result:      "success" | "denied" | "error"
    }

OUTPUT:
  - Audit log queries (by user, resource, action, date range)
  - Compliance reports
  - Usage analytics dashboards

CONNECTIONS:
  Message Queue --> [Audit Service] --> ClickHouse (append-only storage)
                                    --> PostgreSQL (audit query metadata)
  All Services  --> Message Queue (emit audit events)
```

---

## 3. Data Flow Diagrams

### 3.1 Document Upload Flow

```
CLIENT                API GW         DOC SVC        S3/MinIO      POSTGRES
  |                     |               |               |            |
  |-- POST /documents ->|               |               |            |
  |   (multipart file)  |               |               |            |
  |                     |-- validate -->|               |            |
  |                     |   JWT token   |               |            |
  |                     |               |               |            |
  |                     |-- forward --->|               |            |
  |                     |   request     |               |            |
  |                     |               |-- store ----->|            |
  |                     |               |   original    |            |
  |                     |               |<- path -------|            |
  |                     |               |               |            |
  |                     |               |-- generate -->|            |
  |                     |               |   thumbnails  |            |
  |                     |               |               |            |
  |                     |               |-- INSERT -----|----------->|
  |                     |               |   document +  |            |
  |                     |               |   version     |            |
  |                     |               |               |            |
  |                     |               |               |  MESSAGE QUEUE
  |                     |               |-- publish ----|----------->|
  |                     |               |  "doc.process"|            |
  |                     |               |               |            |
  |<-- 202 Accepted ----|<- response ---|               |            |
  |   {doc_id, status:  |               |               |            |
  |    "processing"}    |               |               |            |
  |                     |               |               |            |

                     MESSAGE QUEUE      OCR SVC       INDEXING SVC
                         |               |               |
  (async)                |-- consume -->|               |
                         |  "doc.process"|              |
                         |               |-- OCR ------>|
                         |               |   per page   |
                         |               |               |
                         |               |-- store text->| (to Postgres)
                         |               |               |
                         |               |-- publish --->|
                         |               | "ocr.complete"|
                         |               |               |
                         |               |               |-- index pages
                         |               |               |   (Meilisearch)
                         |               |               |
                         |               |               |-- embed pages
                         |               |               |   (Qdrant)
                         |               |               |
                         |               |               |-- publish
                         |               |               | "index.complete"
                         |               |               |
                         |        POSTGRES UPDATE: status = "ready"
```

### 3.2 Document Search Flow (Keyword vs Semantic)

```
KEYWORD SEARCH:
==============
Client                  Search Svc           Meilisearch        Postgres
  |                        |                     |                  |
  |-- GET /search -------->|                     |                  |
  |   ?q="invoice+15420"   |                     |                  |
  |   &mode=keyword        |                     |                  |
  |   &granularity=line    |                     |                  |
  |                        |-- search "lines" -->|                  |
  |                        |   index with query  |                  |
  |                        |   + ACL filter      |                  |
  |                        |<-- ranked matches --|                  |
  |                        |                     |                  |
  |                        |-- fetch doc meta ---|----------------->|
  |                        |<-- titles, paths ---|------------------|
  |                        |                     |                  |
  |<-- results with -------|                     |                  |
  |    page+line numbers   |                     |                  |
  |    and highlights      |                     |                  |


SEMANTIC SEARCH:
===============
Client                  Search Svc         Embedding API      Qdrant
  |                        |                     |               |
  |-- GET /search -------->|                     |               |
  |   ?q="quarterly cost"  |                     |               |
  |   &mode=semantic       |                     |               |
  |                        |-- embed query ----->|               |
  |                        |<-- query vector ----|               |
  |                        |                     |               |
  |                        |-- vector search ----|-------------->|
  |                        |   with ACL filter   |               |
  |                        |<-- nearest pages ---|---------------|
  |                        |                     |               |
  |<-- results with -------|                     |               |
  |    page numbers and    |                     |               |
  |    relevance scores    |                     |               |


HYBRID SEARCH (Smart Search):
=============================
Client               Search Svc        Meilisearch    Qdrant    Embed API
  |                     |                  |            |           |
  |-- GET /search ----->|                  |            |           |
  |   ?q="payment due"  |                  |            |           |
  |   &mode=smart       |                  |            |           |
  |                     |-- keyword ------>|            |           |
  |                     |   search         |            |           |
  |                     |                  |            |           |
  |                     |-- embed query ---|------------|---------->|
  |                     |<-- vector -------|------------|-----------|
  |                     |                  |            |           |
  |                     |-- vector --------|----------->|           |
  |                     |   search         |            |           |
  |                     |                  |            |           |
  |                     |<-- kw results ---|            |           |
  |                     |<-- vec results --|------------|           |
  |                     |                  |            |           |
  |                     |-- RRF fusion --->|            |           |
  |                     |   (in memory)    |            |           |
  |                     |                  |            |           |
  |<-- fused results ---|                  |            |           |
  |    with page+line   |                  |            |           |
  |    citations        |                  |            |           |
```

### 3.3 Source-First Smart Query Flow (Replaces RAG)

```
Client          Smart Query Svc    Qdrant       Meilisearch    Postgres/S3
  |                 |                |               |               |
  |-- POST -------->|                |               |               |
  |  /api/v1/query  |                |               |               |
  |  "What are the  |                |               |               |
  |   payment terms |                |               |               |
  |   with Acme?"   |                |               |               |
  |                 |                |               |               |
  |                 |== RUN IN PARALLEL ================================|
  |                 |                |               |               |
  |                 |-- embed ------>|               |               |
  |                 |   question     |               |               |
  |                 |-- vector ----->|               |               |
  |                 |   search       |               |               |
  |                 |                |               |               |
  |                 |-- keyword -----|-------------->|               |
  |                 |   search lines |               |               |
  |                 |                |               |               |
  |                 |-- keyword -----|-------------->|               |
  |                 |   search pages |               |               |
  |                 |                |               |               |
  |                 |== COLLECT RESULTS ================================|
  |                 |                |               |               |
  |                 |<- page chunks -|               |               |
  |                 |<- line matches |---------------|               |
  |                 |<- page matches |---------------|               |
  |                 |                |               |               |
  |                 |-- RRF FUSION + DEDUP                          |
  |                 |   (rank and group by document)                |
  |                 |                |               |               |
  |                 |-- fetch context|---------------|-------------->|
  |                 |   (surrounding lines,          |  (from PG)   |
  |                 |    bounding boxes,             |               |
  |                 |    doc metadata)               |               |
  |                 |                |               |               |
  |                 |-- extract facts|               |               |
  |                 |   (rule-based, |               |               |
  |                 |    no LLM)     |               |               |
  |                 |                |               |               |
  |                 |-- build viewer |---------------|-------------->|
  |                 |   URLs with    |               |  (presigned   |
  |                 |   highlight    |               |   from S3)    |
  |                 |   params       |               |               |
  |                 |                |               |               |
  |<-- SOURCE ------|                |               |               |
  |    EVIDENCE     |   NO LLM CALL NEEDED                         |
  |    PACKAGE      |   (original text + locations + viewer URLs)   |
  |                 |                |               |               |
  |                 |                |               |               |
  |  CLIENT RENDERS:                |               |               |
  |  +------------------+---------------------------+               |
  |  | Floating Panel   | Document Viewer           |               |
  |  | - 3 sources      | - PDF/image at page 4     |               |
  |  | - key facts      | - lines 15-20 highlighted |               |
  |  | - [Jump to ➜]    | - yellow highlight boxes  |               |
  |  | - AI summary     | - [Prev] [Next] source    |               |
  |  |   (collapsed,    |                           |               |
  |  |    optional)     |                           |               |
  |  +------------------+---------------------------+               |
```

### 3.4 Workflow Execution Flow

```
User A              Workflow Engine       Postgres       Notification Svc
  |                      |                   |                  |
  |-- submit doc ------->|                   |                  |
  |   for review         |                   |                  |
  |                      |-- validate ------>|                  |
  |                      |   transition      |                  |
  |                      |   (Draft->Submit) |                  |
  |                      |                   |                  |
  |                      |-- UPDATE state -->|                  |
  |                      |   to "Submitted"  |                  |
  |                      |                   |                  |
  |                      |-- notify ---------|----------------->|
  |                      |   reviewers       |                  |
  |                      |                   |        email/in-app to
  |                      |                   |        Reviewer B
  |<-- 200 submitted ----|                   |                  |

Reviewer B           Workflow Engine       Postgres       Notification Svc
  |                      |                   |                  |
  |-- approve doc ------>|                   |                  |
  |   with comments      |                   |                  |
  |                      |-- check: is ------>|                 |
  |                      |   reviewer allowed |                 |
  |                      |   to approve?      |                 |
  |                      |                   |                  |
  |                      |-- check: min ----->|                 |
  |                      |   approvals met?   |                 |
  |                      |   (2 required,     |                 |
  |                      |    this is #1)     |                 |
  |                      |                   |                  |
  |                      |-- UPDATE: add ---->|                 |
  |                      |   approval record  |                 |
  |                      |   (still Under     |                 |
  |                      |    Review)         |                 |
  |                      |                   |                  |

Reviewer C           Workflow Engine       Postgres       Notification Svc
  |                      |                   |                  |
  |-- approve doc ------>|                   |                  |
  |                      |-- check: min ----->|                 |
  |                      |   approvals = 2   |                  |
  |                      |   THRESHOLD MET   |                  |
  |                      |                   |                  |
  |                      |-- UPDATE state -->|                  |
  |                      |   to "Approved"   |                  |
  |                      |                   |                  |
  |                      |-- execute ------->|                  |
  |                      |   actions:        |                  |
  |                      |   - move cabinet  |                  |
  |                      |   - set metadata  |                  |
  |                      |                   |                  |
  |                      |-- notify ---------|----------------->|
  |                      |   owner: approved |         email to User A
  |                      |                   |                  |
```

---

## 4. Data Models

### 4.1 Entity Relationship Overview

```
+------------------+       +-------------------+       +----------------+
|     Cabinet      |<----->|    Document        |<----->|    Tag         |
|------------------|  N:M  |-------------------|  N:M  |----------------|
| id               |       | id                |       | id             |
| name             |       | title             |       | label          |
| description      |       | description       |       | color          |
| parent_id (self) |       | document_type_id  |       | created_at     |
| created_by       |       | cabinet_id        |       +----------------+
| created_at       |       | created_by        |
+------------------+       | created_at        |
                           | current_version   |
                           | workflow_state_id  |
                           +-------------------+
                                  |  1:N
                                  v
                    +----------------------------+
                    |    DocumentVersion          |
                    |----------------------------|
                    | id                          |
                    | document_id                 |
                    | version_number              |
                    | file_path (S3 key)          |
                    | file_name                   |
                    | file_size                   |
                    | mime_type                   |
                    | checksum_sha256             |
                    | page_count                  |
                    | ocr_status                  |
                    | index_status                |
                    | comment                     |
                    | uploaded_by                 |
                    | uploaded_at                 |
                    +----------------------------+
                                  |  1:N
                                  v
                    +----------------------------+
                    |         Page                |
                    |----------------------------|
                    | id                          |
                    | version_id                  |
                    | page_number                 |
                    | full_text                   |
                    | word_count                  |
                    | language                    |
                    | ocr_confidence              |
                    | thumbnail_path              |
                    | width_px                    |
                    | height_px                   |
                    | embedding_id (Qdrant ref)   |
                    +----------------------------+
                                  |  1:N
                                  v
                    +----------------------------+
                    |         Line                |
                    |----------------------------|
                    | id                          |
                    | page_id                     |
                    | line_number                 |
                    | text                        |
                    | bbox_x                      |
                    | bbox_y                      |
                    | bbox_w                      |
                    | bbox_h                      |
                    | confidence                  |
                    +----------------------------+
```

### 4.2 User & Permission Models

```
+------------------+       +-------------------+       +----------------+
|      User        |<----->|    UserRole        |------>|     Role       |
|------------------|  1:N  |-------------------|       |----------------|
| id               |       | user_id           |       | id             |
| email            |       | role_id           |       | name           |
| password_hash    |       | granted_by        |       | description    |
| full_name        |       | granted_at        |       | is_system      |
| avatar_url       |       +-------------------+       +----------------+
| is_active        |                                          |  1:N
| mfa_enabled      |                                          v
| mfa_secret       |                               +-------------------+
| last_login       |                               | RolePermission    |
| created_at       |                               |-------------------|
+------------------+                               | role_id           |
       |                                           | permission_id     |
       | 1:N                                       +-------------------+
       v                                                  |
+-------------------+                              +-------------------+
|      ACLEntry     |                              |   Permission      |
|-------------------|                              |-------------------|
| id                |                              | id                |
| subject_type      | -- "user"|"role"|"group"     | codename          |
| subject_id        |                              | description       |
| resource_type     | -- "document"|"cabinet"      | resource_type     |
| resource_id       |                              +-------------------+
| permission        | -- "read"|"write"|"delete"
| is_inherited      |
| granted_by        |
| granted_at        |
| expires_at        |
+-------------------+
```

### 4.3 Workflow Models

```
+----------------------+        +-------------------------+
|  WorkflowTemplate    |------->|  WorkflowState          |
|----------------------|  1:N   |-------------------------|
| id                   |        | id                      |
| name                 |        | template_id             |
| description          |        | name                    |
| is_active            |        | state_type              |
| created_by           |        |   ("initial"|"intermediate"|"final")
| created_at           |        | actions_on_enter (JSON) |
+----------------------+        | sla_hours               |
                                +-------------------------+
                                       |  1:N
                                       v
                        +-----------------------------+
                        |  WorkflowTransition         |
                        |-----------------------------|
                        | id                          |
                        | from_state_id               |
                        | to_state_id                 |
                        | name                        |
                        | required_role               |
                        | required_approvals          |
                        | conditions (JSON)           |
                        | actions (JSON)              |
                        +-----------------------------+

+-------------------------+       +---------------------------+
| WorkflowInstance        |       | WorkflowAction            |
|-------------------------|       |---------------------------|
| id                      |       | id                        |
| template_id             |       | instance_id               |
| document_id             |       | transition_id             |
| current_state_id        |       | performed_by              |
| started_by              |       | action ("approve"|"reject"|
| started_at              |       |          "comment"|"escalate")
| completed_at            |       | comment                   |
| deadline_at             |       | performed_at              |
+-------------------------+       +---------------------------+
```

### 4.4 Metadata & Document Type Models

```
+----------------------+       +-------------------------+
|  DocumentType        |------>| DocumentTypeMetaField   |
|----------------------| 1:N   |-------------------------|
| id                   |       | id                      |
| name                 |       | document_type_id        |
| description          |       | field_name              |
| icon                 |       | field_type              |
| default_workflow_id  |       |   ("string"|"number"|"date"|
+----------------------+       |    "boolean"|"select"|"multi")
                               | is_required             |
                               | validation_regex        |
                               | select_options (JSON)   |
                               | display_order           |
                               +-------------------------+

+-------------------------+
| DocumentMetadata        |
|-------------------------|
| id                      |
| document_id             |
| field_id                |
| value_string            |
| value_number            |
| value_date              |
| value_boolean           |
| set_by                  |
| set_at                  |
+-------------------------+
```

### 4.5 Complete SQL Schema (PostgreSQL)

```sql
-- Core document tables
CREATE TABLE cabinets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    parent_id       UUID REFERENCES cabinets(id) ON DELETE SET NULL,
    created_by      UUID NOT NULL REFERENCES users(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_cabinets_parent ON cabinets(parent_id);

CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           VARCHAR(500) NOT NULL,
    description     TEXT,
    document_type_id UUID REFERENCES document_types(id),
    cabinet_id      UUID REFERENCES cabinets(id),
    current_version_id UUID,  -- set after first version created
    status          VARCHAR(50) NOT NULL DEFAULT 'processing',
                    -- 'processing', 'ready', 'error', 'archived'
    created_by      UUID NOT NULL REFERENCES users(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at      TIMESTAMPTZ  -- soft delete
);
CREATE INDEX idx_documents_cabinet ON documents(cabinet_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_created ON documents(created_at DESC);

CREATE TABLE document_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version_number  INTEGER NOT NULL,
    file_path       VARCHAR(1000) NOT NULL,  -- S3 key
    file_name       VARCHAR(500) NOT NULL,
    file_size       BIGINT NOT NULL,
    mime_type       VARCHAR(100) NOT NULL,
    checksum_sha256 CHAR(64) NOT NULL,
    page_count      INTEGER,
    ocr_status      VARCHAR(50) DEFAULT 'pending',
                    -- 'pending','processing','completed','failed'
    index_status    VARCHAR(50) DEFAULT 'pending',
    comment         TEXT,
    uploaded_by     UUID NOT NULL REFERENCES users(id),
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(document_id, version_number)
);

CREATE TABLE pages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id      UUID NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
    page_number     INTEGER NOT NULL,
    full_text       TEXT,
    word_count      INTEGER DEFAULT 0,
    language        VARCHAR(10),
    ocr_confidence  REAL,  -- 0.0 to 1.0
    thumbnail_path  VARCHAR(1000),
    width_px        INTEGER,
    height_px       INTEGER,
    embedding_id    VARCHAR(100),  -- reference to Qdrant point ID
    UNIQUE(version_id, page_number)
);
CREATE INDEX idx_pages_version ON pages(version_id);

CREATE TABLE lines (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id         UUID NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    line_number     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    bbox_x          INTEGER,
    bbox_y          INTEGER,
    bbox_w          INTEGER,
    bbox_h          INTEGER,
    confidence      REAL,
    UNIQUE(page_id, line_number)
);
CREATE INDEX idx_lines_page ON lines(page_id);
CREATE INDEX idx_lines_text_trgm ON lines USING gin(text gin_trgm_ops);

-- User and permission tables
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) NOT NULL UNIQUE,
    password_hash   VARCHAR(255),  -- null for OAuth-only users
    full_name       VARCHAR(255) NOT NULL,
    avatar_url      VARCHAR(500),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    mfa_enabled     BOOLEAN NOT NULL DEFAULT FALSE,
    mfa_secret      VARCHAR(100),
    last_login      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE roles (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(100) NOT NULL UNIQUE,
    description     TEXT,
    is_system       BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE permissions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    codename        VARCHAR(100) NOT NULL UNIQUE,  -- e.g. 'document.create'
    description     TEXT,
    resource_type   VARCHAR(50) NOT NULL  -- 'document','cabinet','workflow','system'
);

CREATE TABLE role_permissions (
    role_id         UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    permission_id   UUID NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
    PRIMARY KEY (role_id, permission_id)
);

CREATE TABLE user_roles (
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id         UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    granted_by      UUID REFERENCES users(id),
    granted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, role_id)
);

CREATE TABLE acl_entries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_type    VARCHAR(20) NOT NULL,  -- 'user','role','group'
    subject_id      UUID NOT NULL,
    resource_type   VARCHAR(20) NOT NULL,  -- 'document','cabinet'
    resource_id     UUID NOT NULL,
    permission      VARCHAR(20) NOT NULL,  -- 'read','write','delete','share','full'
    is_inherited    BOOLEAN NOT NULL DEFAULT FALSE,
    granted_by      UUID REFERENCES users(id),
    granted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,
    UNIQUE(subject_type, subject_id, resource_type, resource_id, permission)
);
CREATE INDEX idx_acl_resource ON acl_entries(resource_type, resource_id);
CREATE INDEX idx_acl_subject ON acl_entries(subject_type, subject_id);

-- Tags
CREATE TABLE tags (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label           VARCHAR(100) NOT NULL UNIQUE,
    color           VARCHAR(7) DEFAULT '#6366f1',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE document_tags (
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag_id          UUID NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (document_id, tag_id)
);

-- Workflow tables
CREATE TABLE workflow_templates (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_by      UUID NOT NULL REFERENCES users(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE workflow_states (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id     UUID NOT NULL REFERENCES workflow_templates(id) ON DELETE CASCADE,
    name            VARCHAR(100) NOT NULL,
    state_type      VARCHAR(20) NOT NULL,  -- 'initial','intermediate','final'
    actions_on_enter JSONB,
    sla_hours       INTEGER
);

CREATE TABLE workflow_transitions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_state_id   UUID NOT NULL REFERENCES workflow_states(id),
    to_state_id     UUID NOT NULL REFERENCES workflow_states(id),
    name            VARCHAR(100) NOT NULL,
    required_role   UUID REFERENCES roles(id),
    required_approvals INTEGER DEFAULT 1,
    conditions      JSONB,
    actions         JSONB
);

CREATE TABLE workflow_instances (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id     UUID NOT NULL REFERENCES workflow_templates(id),
    document_id     UUID NOT NULL REFERENCES documents(id),
    current_state_id UUID NOT NULL REFERENCES workflow_states(id),
    started_by      UUID NOT NULL REFERENCES users(id),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    deadline_at     TIMESTAMPTZ
);
CREATE INDEX idx_wf_instances_doc ON workflow_instances(document_id);

CREATE TABLE workflow_actions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instance_id     UUID NOT NULL REFERENCES workflow_instances(id),
    transition_id   UUID REFERENCES workflow_transitions(id),
    performed_by    UUID NOT NULL REFERENCES users(id),
    action          VARCHAR(50) NOT NULL,
    comment         TEXT,
    performed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Metadata
CREATE TABLE document_types (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL UNIQUE,
    description     TEXT,
    icon            VARCHAR(50),
    default_workflow_id UUID REFERENCES workflow_templates(id)
);

CREATE TABLE document_type_meta_fields (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_type_id UUID NOT NULL REFERENCES document_types(id) ON DELETE CASCADE,
    field_name      VARCHAR(100) NOT NULL,
    field_type      VARCHAR(20) NOT NULL,
    is_required     BOOLEAN NOT NULL DEFAULT FALSE,
    validation_regex VARCHAR(500),
    select_options  JSONB,
    display_order   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE document_metadata (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    field_id        UUID NOT NULL REFERENCES document_type_meta_fields(id),
    value_string    TEXT,
    value_number    DOUBLE PRECISION,
    value_date      DATE,
    value_boolean   BOOLEAN,
    set_by          UUID REFERENCES users(id),
    set_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(document_id, field_id)
);
```

---

## 5. Search Architecture Deep Dive

### 5.1 OCR Text to Searchable Chunks

```
DOCUMENT (PDF, 10 pages)
          |
          v
   +------+------+
   |  OCR Engine  |  (Tesseract with HOCR output)
   +------+------+
          |
          v
   RAW OCR OUTPUT (per page):
   +--------------------------------------------------+
   | Page 1:                                           |
   |   Line 1: "INVOICE" (bbox: 200,50,300,70)        |
   |   Line 2: "Invoice #: INV-2024-042" (bbox: ...)  |
   |   Line 3: "Date: January 15, 2024" (bbox: ...)   |
   |   Line 4: "" (empty line)                         |
   |   Line 5: "Bill To:" (bbox: ...)                  |
   |   Line 6: "Acme Corporation" (bbox: ...)          |
   |   ...                                             |
   +--------------------------------------------------+
          |
          v
   THREE LEVELS OF INDEXING:

   LEVEL 1: DOCUMENT-LEVEL (for overview search)
   +--------------------------------------------------+
   | Concatenate all pages -> single document text     |
   | Store as: full_document_text                      |
   | Use for: broad relevance ranking                  |
   +--------------------------------------------------+

   LEVEL 2: PAGE-LEVEL (for page-precision results)
   +--------------------------------------------------+
   | Each page is one search unit                      |
   | Stored in: Meilisearch "pages" index              |
   |            Qdrant "page_embeddings" collection     |
   | Use for: "which page discusses X?"                |
   |                                                   |
   | Embedding: full page text -> 1536-dim vector      |
   | If page > 1024 tokens: split with 64-token overlap|
   +--------------------------------------------------+

   LEVEL 3: LINE-LEVEL (for exact precision results)
   +--------------------------------------------------+
   | Each line is one search unit                      |
   | Stored in: Meilisearch "lines" index              |
   |            PostgreSQL "lines" table (with bbox)   |
   | Use for: "find exact line mentioning $15,420"     |
   |                                                   |
   | NOT embedded individually (too granular for       |
   | semantic search - would lose context)             |
   +--------------------------------------------------+
```

### 5.2 Vector Embedding Pipeline

```
EMBEDDING GENERATION FLOW:

Page Text (from OCR)
    |
    v
+-------------------+
| Pre-processing    |
| - Clean OCR noise |
| - Normalize ws    |
| - Remove headers/ |
|   footers if      |
|   repeated        |
+-------------------+
    |
    v
+-------------------+
| Token Counter     |
| (tiktoken)        |
+-------------------+
    |
    +-- < 512 tokens ---------> Embed as single chunk
    |
    +-- 512-1024 tokens ------> Embed as single chunk
    |
    +-- > 1024 tokens --------> Split into overlapping chunks:
                                 Chunk 1: tokens[0:512]
                                 Chunk 2: tokens[448:960]  (64 overlap)
                                 Chunk 3: tokens[896:1408]
                                 ... each chunk embedded separately
    |
    v
+-------------------+
| Embedding Model   |
| OpenAI text-      |
| embedding-3-small |
| (1536 dimensions) |
|                   |
| OR local:         |
| all-MiniLM-L6-v2  |
| (384 dimensions)  |
+-------------------+
    |
    v
+-------------------+
| Qdrant Upsert     |
|                   |
| Point {           |
|   id: page_uuid,  |
|   vector: [...],  |
|   payload: {      |
|     document_id,  |
|     page_number,  |
|     chunk_index,  |
|     text_preview, |
|     acl_read: [], |
|     cabinet,      |
|     tags,         |
|     uploaded_at   |
|   }               |
| }                 |
+-------------------+

EMBEDDING COST ESTIMATE (OpenAI text-embedding-3-small):
  50,000 pages x 2,500 chars avg = 125M chars = ~31M tokens
  Price: $0.02 per 1M tokens
  Total: 31 x $0.02 = $0.62 for initial indexing

  For local model (all-MiniLM-L6-v2 via sentence-transformers):
  Cost: $0 (runs on CPU/GPU locally)
  Speed: ~100 pages/minute on M1 Mac
  Total time: ~8.3 minutes for 50K pages
```

### 5.3 How RAG Queries Work End-to-End

```
USER QUESTION: "What is the warranty period for products sold to Acme?"

STEP 1: QUERY EMBEDDING
+---------------------------------------------------------------+
| Input:  "What is the warranty period for products sold to Acme"|
| Model:  text-embedding-3-small                                |
| Output: [0.023, -0.041, 0.089, ...]  (1536 floats)            |
| Time:   ~200ms                                                 |
+---------------------------------------------------------------+

STEP 2: VECTOR RETRIEVAL (Qdrant)
+---------------------------------------------------------------+
| Request:                                                       |
|   collection: "page_embeddings"                                |
|   vector: [0.023, -0.041, ...]                                 |
|   limit: 20                                                    |
|   score_threshold: 0.65                                        |
|   filter: {                                                    |
|     must: [{                                                   |
|       key: "acl_read",                                         |
|       match: { any: ["user_123", "role_editor"] }              |
|     }]                                                         |
|   }                                                            |
|                                                                |
| Response (top 5 shown):                                        |
|   1. score=0.91, doc="Acme Master Agreement", pg=12            |
|      "...warranty period of 24 months from delivery..."        |
|   2. score=0.87, doc="Acme Product Addendum", pg=3             |
|      "...extended warranty available for 36 months..."         |
|   3. score=0.82, doc="Acme Master Agreement", pg=13            |
|      "...warranty does not cover damage from misuse..."        |
|   4. score=0.78, doc="Standard Warranty Policy", pg=1          |
|      "...default warranty period is 12 months..."              |
|   5. score=0.72, doc="Acme Purchase Order 2024-Q1", pg=2       |
|      "...subject to warranty terms in master agreement..."     |
| Time: ~50ms                                                    |
+---------------------------------------------------------------+

STEP 3: KEYWORD AUGMENTATION (Meilisearch)
+---------------------------------------------------------------+
| Query: "warranty period Acme products"                         |
| Index: "pages"                                                 |
| Filter: acl_read IN [user_123, role_editor]                    |
|                                                                |
| Additional hits not in vector results:                         |
|   - doc="Acme Amendment 2024", pg=5                            |
|     "warranty period amended to 18 months for Category B"      |
| Time: ~20ms                                                    |
+---------------------------------------------------------------+

STEP 4: CONTEXT ASSEMBLY
+---------------------------------------------------------------+
| Deduplicate and rank all retrieved chunks                      |
| Select top 5 most relevant (by RRF score)                      |
| For each: fetch full page text from PostgreSQL                 |
| Also fetch surrounding lines for context                       |
|                                                                |
| Assemble prompt:                                               |
|   System: "You are a document Q&A assistant..."                |
|   Sources: [Source 1] through [Source 5] with full text         |
|   Question: original user question                             |
| Total context: ~3,000 tokens                                   |
+---------------------------------------------------------------+

STEP 5: LLM GENERATION
+---------------------------------------------------------------+
| Model: GPT-4o (or Claude 3.5 Sonnet)                           |
| Input: ~3,500 tokens (system + context + question)             |
| Output: ~300 tokens                                            |
| Time: ~2-3 seconds                                             |
|                                                                |
| Generated Answer:                                              |
| "The warranty period for products sold to Acme depends on      |
|  the product category:                                         |
|                                                                |
|  - Standard products: 24 months from delivery date [Source 1]  |
|  - Category B products: 18 months (amended in 2024) [Source 5] |
|  - Extended warranty option: 36 months available [Source 2]    |
|                                                                |
|  The base company warranty policy is 12 months [Source 4],     |
|  but the Acme master agreement specifies longer terms.         |
|  Note that warranty does not cover damage from misuse          |
|  [Source 3]."                                                  |
+---------------------------------------------------------------+

STEP 6: CITATION RESOLUTION
+---------------------------------------------------------------+
| Map each [Source N] to exact document location:                 |
|                                                                |
| Citation 1: {                                                  |
|   document: "Acme Master Agreement",                           |
|   document_id: "uuid-aaa",                                     |
|   page: 12,                                                    |
|   lines: [8, 12],                                              |
|   quote: "warranty period of 24 months from delivery"          |
| }                                                              |
| Citation 5: {                                                  |
|   document: "Acme Amendment 2024",                             |
|   document_id: "uuid-eee",                                     |
|   page: 5,                                                     |
|   lines: [3, 5],                                               |
|   quote: "warranty period amended to 18 months for Cat B"      |
| }                                                              |
| ... etc                                                        |
+---------------------------------------------------------------+

TOTAL LATENCY: ~3-4 seconds
  Embedding:    200ms
  Qdrant:       50ms
  Meilisearch:  20ms
  DB lookups:   30ms
  LLM:          2-3s
  Processing:   100ms
```

### 5.4 ACL-Filtered Search

```
PROBLEM: User must only see search results for documents they can access.

SOLUTION: Pre-compute ACL lists and store them in search indexes.

ON DOCUMENT CREATE / ACL CHANGE:
  1. Compute effective_readers for the document:
     - All users with explicit "read" ACL on this document
     - All users whose role has "document.read" permission
     - All users in groups with "read" ACL on this document
     - All users with "read" on parent cabinet (inherited)

  2. Store as array in Meilisearch and Qdrant:
     acl_read: ["user_123", "user_456", "role_editor", "group_finance"]

  3. On ACL change: re-compute and update all affected indexes

ON SEARCH:
  1. Determine user's access identifiers:
     access_ids = [user_id, ...user_role_ids, ...user_group_ids]

  2. Add filter to Meilisearch query:
     filter: "acl_read IN [user_123, role_editor, group_finance]"

  3. Add filter to Qdrant query:
     filter: { must: [{ key: "acl_read", match: { any: access_ids } }] }

TRADE-OFF:
  PRO: Search is fast (no join with ACL table at query time)
  CON: ACL changes require re-indexing affected documents
  MITIGATION: ACL changes are infrequent; queue re-indexing async
```

---

## 6. Storage Architecture

### 6.1 Multi-Tier Storage

```
+-------------------------------------------------------------------+
|                     STORAGE ARCHITECTURE                          |
+-------------------------------------------------------------------+

HOT TIER (frequently accessed, low latency)
+-------------------------------------------------------------------+
| Redis Cache          | Recently accessed documents, search results |
| (in-memory)          | Session data, user preferences              |
| TTL: 15min-1hr       | Embedding query cache                       |
| Size: 2-8 GB         | Workflow state cache                        |
+-------------------------------------------------------------------+
                              |
                              v
WARM TIER (active data, moderate latency)
+-------------------------------------------------------------------+
| PostgreSQL           | All metadata, ACLs, users, workflows        |
| (SSD storage)        | Page text, line text, document records       |
|                      | Full ACID, complex queries                  |
| Meilisearch          | Full-text search indexes (pages, lines)     |
| (SSD storage)        | Typo-tolerant, faceted search               |
|                      |                                             |
| Qdrant               | Vector embeddings for semantic search       |
| (SSD + memory-mapped)| HNSW index for fast nearest-neighbor        |
|                      |                                             |
| MinIO / S3           | Original files, thumbnails, OCR text files  |
| (SSD or HDD)        | Current and recent versions                 |
+-------------------------------------------------------------------+
                              |
                              v
COLD TIER (archival, high latency acceptable)
+-------------------------------------------------------------------+
| S3 Glacier / Tape    | Old document versions (>1 year)             |
| or Cold MinIO tier   | Archived documents                          |
|                      | Audit logs older than retention period       |
| ClickHouse           | Historical audit logs, analytics data       |
| (HDD storage)       | Compressed columnar storage                 |
+-------------------------------------------------------------------+
```

### 6.2 Database Connection Architecture

```
APPLICATION SERVERS
     |
     v
+-------------------+
| Connection Pool   |   (pgBouncer or built-in pool)
| Max: 20 per node  |
| Idle timeout: 30s |
+-------------------+
     |
     +-----> PostgreSQL Primary (writes)
     |         |
     |         +-----> Replica 1 (reads - search metadata)
     |         +-----> Replica 2 (reads - API queries)
     |
     +-----> Redis Cluster
     |         |
     |         +-----> Shard 1 (sessions, cache)
     |         +-----> Shard 2 (queues, pub/sub)
     |
     +-----> Qdrant
     |         |
     |         +-----> Collection: page_embeddings
     |         +-----> Collection: query_cache
     |
     +-----> Meilisearch
     |         |
     |         +-----> Index: pages
     |         +-----> Index: lines
     |         +-----> Index: documents
     |
     +-----> MinIO
               |
               +-----> Bucket: edms-originals
               +-----> Bucket: edms-thumbnails
               +-----> Bucket: edms-ocr-text
```

### 6.3 Cache Strategy

```
CACHE LAYERS AND POLICIES:

1. SEARCH RESULT CACHE (Redis)
   Key:    search:{hash(query+filters+user_acl_hash)}
   Value:  serialized search results
   TTL:    5 minutes
   Policy: Cache-aside, invalidate on document index update
   Reason: Identical searches within short window are common

2. DOCUMENT METADATA CACHE (Redis)
   Key:    doc:{document_id}:meta
   Value:  { title, cabinet, tags, current_version, status }
   TTL:    30 minutes
   Policy: Write-through (update cache on DB write)
   Reason: Metadata fetched on every search result display

3. ACL RESOLUTION CACHE (Redis)
   Key:    acl:{user_id}:access_ids
   Value:  [user_id, role_ids..., group_ids...]
   TTL:    15 minutes
   Policy: Cache-aside, invalidate on role/group change
   Reason: ACL check on every search query and document access

4. EMBEDDING QUERY CACHE (Redis)
   Key:    embed:{hash(query_text)}
   Value:  [1536 floats as binary]
   TTL:    1 hour
   Policy: Cache-aside
   Reason: Same question re-embedded is wasteful (200ms saved)

5. THUMBNAIL CACHE (CDN or Nginx proxy_cache)
   Key:    /thumbnails/{doc_id}/{page}/{size}.jpg
   TTL:    24 hours
   Policy: Cache-aside with stale-while-revalidate
   Reason: Thumbnails are immutable per version, heavily accessed

6. PRESIGNED URL CACHE (Redis)
   Key:    presigned:{doc_id}:{version}:{user_id}
   Value:  presigned S3 URL
   TTL:    10 minutes (shorter than S3 URL expiry of 15min)
   Policy: Cache-aside
   Reason: Avoid regenerating presigned URLs on repeated access
```

---

## 7. Scalability & Performance

### 7.1 Scaling Strategy by Component

```
+-------------------------------------------------------------------+
|              HORIZONTAL SCALING STRATEGY                          |
+-------------------------------------------------------------------+

COMPONENT          | SCALING METHOD          | TRIGGER METRIC
-------------------|-------------------------|------------------------
API Gateway        | Add more instances       | CPU > 70%, latency > 100ms
                   | (stateless, behind LB)  |
                   |                         |
Auth Service       | Horizontal (stateless)  | Auth QPS > 500/instance
                   | JWT = no shared state   |
                   |                         |
Document Service   | Horizontal (stateless)  | Upload QPS > 50/instance
                   |                         |
Search Service     | Horizontal (stateless)  | Search QPS > 100/instance
                   |                         |
OCR Workers        | Horizontal (add workers)| Queue depth > 100 jobs
                   | CPU-bound, scale by CPU | Processing time > 5min
                   |                         |
Embedding Workers  | Horizontal (add workers)| Queue depth > 500 chunks
                   | GPU if available        | If using local model
                   |                         |
PostgreSQL         | Vertical first, then    | Query latency > 50ms
                   | read replicas           | Connections > 80% pool
                   | Shard at 100M+ rows     |
                   |                         |
Meilisearch        | Vertical (single node   | Index size > 50GB
                   | handles 10M+ docs)      | Search latency > 100ms
                   | Multi-node at 100M+     |
                   |                         |
Qdrant             | Vertical, then sharding | Collection > 10M vectors
                   | Built-in clustering     | Search latency > 100ms
                   |                         |
Redis              | Cluster mode (sharding) | Memory > 80% capacity
                   |                         |
MinIO/S3           | Add nodes to cluster    | Storage > 80% capacity
                   | (MinIO distributed mode)|
```

### 7.2 Queue-Based Async Processing

```
MESSAGE QUEUE ARCHITECTURE (BullMQ on Redis):

QUEUE: document.upload
  Producer: Document Service
  Consumer: Upload Workers (2-4 instances)
  Job: { document_id, version_id, file_path }
  Concurrency: 5 per worker
  Retry: 3 attempts, exponential backoff

QUEUE: document.ocr
  Producer: Upload Workers
  Consumer: OCR Workers (2-8 instances, CPU-intensive)
  Job: { document_id, version_id, page_number, image_path }
  Concurrency: 2 per worker (CPU-bound)
  Retry: 2 attempts
  Timeout: 60 seconds per page

QUEUE: document.index
  Producer: OCR Workers
  Consumer: Indexing Workers (2-4 instances)
  Job: { document_id, version_id, page_id, text }
  Concurrency: 10 per worker
  Retry: 3 attempts

QUEUE: document.embed
  Producer: Indexing Workers
  Consumer: Embedding Workers (1-4 instances)
  Job: { page_id, text, metadata }
  Concurrency: 20 per worker (IO-bound API calls)
  Retry: 3 attempts with backoff
  Rate limit: 3000 requests/min (OpenAI limit)

QUEUE: notifications
  Producer: Workflow Engine, Document Service
  Consumer: Notification Workers (1-2 instances)
  Job: { type, recipient_id, channel, subject, body }
  Concurrency: 10 per worker

QUEUE: audit
  Producer: All services
  Consumer: Audit Workers (1-2 instances)
  Job: { event_type, actor, resource, details, timestamp }
  Concurrency: 50 per worker (fast ClickHouse inserts)

FLOW VISUALIZATION:
  Upload -> [document.upload] -> [document.ocr] -> [document.index]
                                                 -> [document.embed]
                                                 -> [notifications]
  All    -> [audit]
```

### 7.3 Performance Optimization Targets

```
LATENCY TARGETS:

Operation                    | Target (p95) | Strategy
-----------------------------|--------------|---------------------------
Document upload (response)   | < 2s         | Async processing, return 202
Search (keyword)             | < 200ms      | Meilisearch (in-memory index)
Search (semantic)            | < 500ms      | Qdrant HNSW + embed cache
Search (hybrid/smart)        | < 800ms      | Parallel queries + RRF
RAG question answering       | < 5s         | Stream response, cache
Document download            | < 1s         | Presigned URLs, CDN
Thumbnail load               | < 200ms      | CDN cache, pre-generated
Page view (with text)        | < 300ms      | Redis cache + presigned URL
ACL check                    | < 10ms       | Redis-cached access lists
Login                        | < 500ms      | bcrypt(10 rounds) + JWT

THROUGHPUT TARGETS (per instance):
  API Gateway:      1000 req/s
  Search Service:   100 req/s
  Document Service: 50 req/s
  OCR Worker:       30 pages/min (CPU-dependent)
  Embedding Worker: 200 pages/min (API-dependent)
```

---

## 8. Security Architecture

### 8.1 Authentication Flow

```
+-------------------------------------------------------------------+
|                    AUTHENTICATION ARCHITECTURE                    |
+-------------------------------------------------------------------+

FLOW 1: EMAIL/PASSWORD LOGIN
  Client                    Auth Service               PostgreSQL
    |                           |                          |
    |-- POST /auth/login ------>|                          |
    |   { email, password }     |                          |
    |                           |-- SELECT user by email ->|
    |                           |<-- user record ----------|
    |                           |                          |
    |                           |-- bcrypt.compare(        |
    |                           |   password, hash)        |
    |                           |                          |
    |                           |-- if MFA enabled:        |
    |<-- 200 { mfa_required, -->|   return mfa challenge   |
    |    challenge_token }      |                          |
    |                           |                          |
    |-- POST /auth/mfa -------->|                          |
    |   { challenge_token,      |                          |
    |     totp_code }           |-- verify TOTP            |
    |                           |                          |
    |<-- 200 { access_token, ---|                          |
    |    refresh_token }        |                          |

FLOW 2: OAUTH2 LOGIN (e.g., Google)
  Client          Auth Service        Google Auth       PostgreSQL
    |                 |                    |                |
    |-- GET /auth/ -->|                    |                |
    |   oauth/google  |                    |                |
    |                 |-- redirect ------->|                |
    |                 |   (auth URL +      |                |
    |                 |    PKCE challenge)  |                |
    |                 |                    |                |
    |<-- redirect ----|<-- code + state ---|                |
    |   to callback   |                    |                |
    |                 |-- exchange code -->|                |
    |                 |<-- tokens ---------|                |
    |                 |                    |                |
    |                 |-- extract profile  |                |
    |                 |-- upsert user -----|--------------->|
    |                 |                    |                |
    |<-- 200 tokens --|                    |                |

TOKEN STRUCTURE:
  Access Token (JWT, 15-minute expiry):
  {
    "sub": "user_uuid",
    "email": "user@example.com",
    "roles": ["editor", "finance_group"],
    "permissions": ["document.read", "document.create", ...],
    "iat": 1700000000,
    "exp": 1700000900,
    "iss": "edms-auth"
  }

  Refresh Token (opaque, 7-day expiry):
  - Stored hashed in PostgreSQL
  - Rotated on each use (old token invalidated)
  - Bound to device/IP fingerprint
```

### 8.2 Authorization Architecture

```
AUTHORIZATION CHECK FLOW:

Every API request goes through this pipeline:

  Request --> [1. JWT Validation] --> [2. RBAC Check] --> [3. ACL Check]
                    |                      |                    |
                    v                      v                    v
              Verify signature       Check role has       Check user has
              Check expiry           required permission  access to specific
              Extract claims         for this endpoint    resource
                    |                      |                    |
                    v                      v                    v
              401 if invalid         403 if no role       403 if no ACL
                                     permission           entry

IMPLEMENTATION (Express middleware):

  // Middleware chain for a protected endpoint
  router.get('/documents/:id',
    authenticate,          // Step 1: JWT validation
    authorize('document.read'),  // Step 2: RBAC check
    checkDocumentACL('read'),    // Step 3: Object-level ACL
    documentController.get
  );

ACL RESOLUTION ALGORITHM:

  function canAccess(userId, resourceType, resourceId, permission):
    // 1. Check explicit DENY
    deny = findACL(userId, resourceType, resourceId, 'deny', permission)
    if deny: return false

    // 2. Check explicit ALLOW
    allow = findACL(userId, resourceType, resourceId, 'allow', permission)
    if allow: return true

    // 3. Check role-based ALLOW
    userRoles = getUserRoles(userId)
    for role in userRoles:
      allow = findACL(role.id, resourceType, resourceId, 'allow', permission)
      if allow: return true

    // 4. Check inherited from parent cabinet
    if resourceType == 'document':
      cabinetId = getDocumentCabinet(resourceId)
      return canAccess(userId, 'cabinet', cabinetId, permission)

    // 5. Default deny
    return false
```

### 8.3 Encryption

```
ENCRYPTION AT REST:
  - PostgreSQL: Transparent Data Encryption (TDE) or disk-level encryption
  - MinIO/S3: Server-Side Encryption (SSE-S3 or SSE-KMS)
  - Redis: RDB/AOF on encrypted volume
  - Qdrant: Disk-level encryption
  - Backups: AES-256 encrypted before transfer

ENCRYPTION IN TRANSIT:
  - All external traffic: TLS 1.3 (terminated at API Gateway)
  - Internal service-to-service: mTLS (mutual TLS via service mesh)
  - Database connections: TLS required (sslmode=require)
  - S3 connections: HTTPS only

KEY MANAGEMENT:
  - JWT signing: RS256 with RSA-2048 key pair
  - Key rotation: every 90 days, old keys valid for verification only
  - Secrets stored in: HashiCorp Vault or AWS Secrets Manager
  - Environment variables for non-sensitive config
  - NEVER in source code or Docker images
```

### 8.4 Audit Trail

```
AUDIT EVENTS CAPTURED:

Category          | Events
------------------|-----------------------------------------------
Authentication    | login, logout, login_failed, mfa_challenge,
                  | password_change, api_key_created
Authorization     | access_denied, permission_granted, role_changed
Documents         | created, viewed, downloaded, updated, deleted,
                  | version_created, shared, unshared
Search            | query_executed (with query text, result count)
Workflows         | started, transitioned, approved, rejected,
                  | escalated, completed
Admin             | user_created, user_disabled, config_changed,
                  | cabinet_created, tag_created

AUDIT RECORD (ClickHouse):
  CREATE TABLE audit_log (
      event_id    UUID,
      timestamp   DateTime64(3),
      actor_id    UUID,
      actor_email String,
      actor_ip    IPv4,
      action      LowCardinality(String),
      resource_type LowCardinality(String),
      resource_id UUID,
      details     String,  -- JSON
      result      LowCardinality(String),  -- 'success','denied','error'
      user_agent  String
  ) ENGINE = MergeTree()
  ORDER BY (timestamp, actor_id)
  TTL timestamp + INTERVAL 7 YEAR;

QUERY EXAMPLES:
  -- Who accessed document X in the last 30 days?
  SELECT actor_email, action, timestamp
  FROM audit_log
  WHERE resource_id = 'doc-uuid'
    AND timestamp > now() - INTERVAL 30 DAY
  ORDER BY timestamp DESC;

  -- Failed login attempts by IP
  SELECT actor_ip, count() as attempts
  FROM audit_log
  WHERE action = 'login_failed'
    AND timestamp > now() - INTERVAL 1 HOUR
  GROUP BY actor_ip
  HAVING attempts > 5;
```

---

## 9. Deployment Architecture

### 9.1 Docker Compose (Development)

```yaml
# docker-compose.yml
version: '3.8'

services:
  # API Gateway
  traefik:
    image: traefik:v3.0
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"  # dashboard
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./config/traefik:/etc/traefik

  # Application Services
  auth-service:
    build: ./src/services/auth
    environment:
      - DATABASE_URL=postgresql://edms:pass@postgres:5432/edms
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET_FILE=/run/secrets/jwt_secret
    depends_on: [postgres, redis]
    labels:
      - "traefik.http.routers.auth.rule=PathPrefix(`/api/v1/auth`)"

  document-service:
    build: ./src/services/document
    environment:
      - DATABASE_URL=postgresql://edms:pass@postgres:5432/edms
      - S3_ENDPOINT=http://minio:9000
      - S3_BUCKET=edms-originals
      - REDIS_URL=redis://redis:6379
    depends_on: [postgres, redis, minio]
    labels:
      - "traefik.http.routers.docs.rule=PathPrefix(`/api/v1/documents`)"

  search-service:
    build: ./src/services/search
    environment:
      - MEILISEARCH_URL=http://meilisearch:7700
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
    depends_on: [meilisearch, qdrant, redis]
    labels:
      - "traefik.http.routers.search.rule=PathPrefix(`/api/v1/search`)"

  rag-service:
    build: ./src/services/rag
    environment:
      - QDRANT_URL=http://qdrant:6333
      - MEILISEARCH_URL=http://meilisearch:7700
      - OPENAI_API_KEY_FILE=/run/secrets/openai_key
      - REDIS_URL=redis://redis:6379
    depends_on: [qdrant, meilisearch, redis]
    labels:
      - "traefik.http.routers.rag.rule=PathPrefix(`/api/v1/ai`)"

  workflow-service:
    build: ./src/services/workflow
    environment:
      - DATABASE_URL=postgresql://edms:pass@postgres:5432/edms
      - REDIS_URL=redis://redis:6379
    depends_on: [postgres, redis]
    labels:
      - "traefik.http.routers.wf.rule=PathPrefix(`/api/v1/workflows`)"

  # Workers
  ocr-worker:
    build: ./src/workers/ocr
    environment:
      - DATABASE_URL=postgresql://edms:pass@postgres:5432/edms
      - S3_ENDPOINT=http://minio:9000
      - REDIS_URL=redis://redis:6379
    depends_on: [postgres, redis, minio]
    deploy:
      replicas: 2

  indexing-worker:
    build: ./src/workers/indexing
    environment:
      - DATABASE_URL=postgresql://edms:pass@postgres:5432/edms
      - MEILISEARCH_URL=http://meilisearch:7700
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
    depends_on: [postgres, meilisearch, qdrant, redis]

  notification-worker:
    build: ./src/workers/notification
    environment:
      - DATABASE_URL=postgresql://edms:pass@postgres:5432/edms
      - REDIS_URL=redis://redis:6379
      - SMTP_HOST=mailhog
    depends_on: [postgres, redis]

  # Data Stores
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_DB=edms
      - POSTGRES_USER=edms
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"

  meilisearch:
    image: getmeili/meilisearch:v1.6
    environment:
      - MEILI_MASTER_KEY=dev-master-key
    volumes:
      - meili_data:/meili_data
    ports:
      - "7700:7700"

  qdrant:
    image: qdrant/qdrant:v1.7
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"

  clickhouse:
    image: clickhouse/clickhouse-server:23
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    ports:
      - "8123:8123"

  # Dev tools
  mailhog:
    image: mailhog/mailhog
    ports:
      - "1025:1025"
      - "8025:8025"

volumes:
  postgres_data:
  redis_data:
  minio_data:
  meili_data:
  qdrant_data:
  clickhouse_data:
```

### 9.2 Kubernetes Production Architecture

```
NAMESPACE: edms-production

+-------------------------------------------------------------------+
|                    KUBERNETES CLUSTER                              |
+-------------------------------------------------------------------+
|                                                                   |
|  INGRESS CONTROLLER (Traefik / NGINX Ingress)                     |
|  +-------------------------------------------------------------+ |
|  | TLS termination, routing, rate limiting                      | |
|  | cert-manager for auto Let's Encrypt certificates             | |
|  +-------------------------------------------------------------+ |
|                              |                                    |
|  SERVICE MESH (Linkerd / Istio - optional)                        |
|  +-------------------------------------------------------------+ |
|  | mTLS between services, observability, traffic management     | |
|  +-------------------------------------------------------------+ |
|                              |                                    |
|  APPLICATION DEPLOYMENTS                                          |
|  +------------------+ +------------------+ +------------------+  |
|  | auth-service     | | document-service | | search-service   |  |
|  | replicas: 2      | | replicas: 2      | | replicas: 3      |  |
|  | cpu: 250m-500m   | | cpu: 500m-1000m  | | cpu: 500m-1000m  |  |
|  | mem: 256Mi-512Mi | | mem: 512Mi-1Gi   | | mem: 512Mi-1Gi   |  |
|  | HPA: 2-5         | | HPA: 2-6         | | HPA: 3-10        |  |
|  +------------------+ +------------------+ +------------------+  |
|  +------------------+ +------------------+ +------------------+  |
|  | rag-service      | | workflow-service | | notification-svc |  |
|  | replicas: 2      | | replicas: 2      | | replicas: 1      |  |
|  | cpu: 500m-1000m  | | cpu: 250m-500m   | | cpu: 100m-250m   |  |
|  | mem: 1Gi-2Gi     | | mem: 256Mi-512Mi | | mem: 128Mi-256Mi |  |
|  +------------------+ +------------------+ +------------------+  |
|                                                                   |
|  WORKER DEPLOYMENTS                                               |
|  +------------------+ +------------------+ +------------------+  |
|  | ocr-worker       | | indexing-worker  | | audit-worker     |  |
|  | replicas: 3      | | replicas: 2      | | replicas: 1      |  |
|  | cpu: 1000m-2000m | | cpu: 500m-1000m  | | cpu: 100m-250m   |  |
|  | mem: 1Gi-2Gi     | | mem: 512Mi-1Gi   | | mem: 128Mi-256Mi |  |
|  | HPA: 3-10 (CPU)  | | HPA: 2-5 (queue) | |                  |  |
|  +------------------+ +------------------+ +------------------+  |
|                                                                   |
|  STATEFUL SETS                                                    |
|  +------------------+ +------------------+ +------------------+  |
|  | PostgreSQL       | | Redis            | | Qdrant           |  |
|  | primary + 2      | | sentinel mode    | | replicas: 3      |  |
|  | replicas         | | 3 nodes          | | shards: 1-3      |  |
|  | PVC: 100Gi SSD   | | PVC: 16Gi SSD    | | PVC: 50Gi SSD    |  |
|  +------------------+ +------------------+ +------------------+  |
|  +------------------+ +------------------+                       |
|  | Meilisearch      | | ClickHouse       |                       |
|  | replicas: 1      | | replicas: 1      |                       |
|  | PVC: 50Gi SSD    | | PVC: 200Gi HDD   |                       |
|  +------------------+ +------------------+                       |
|                                                                   |
|  EXTERNAL SERVICES                                                |
|  +------------------+                                             |
|  | MinIO (or S3)    |  Operator-managed or external AWS S3        |
|  +------------------+                                             |
|                                                                   |
|  MONITORING STACK                                                 |
|  +------------------+ +------------------+ +------------------+  |
|  | Prometheus       | | Grafana          | | Loki             |  |
|  | (metrics)        | | (dashboards)     | | (logs)           |  |
|  +------------------+ +------------------+ +------------------+  |
+-------------------------------------------------------------------+
```

### 9.3 CI/CD Pipeline

```
PIPELINE STAGES:

1. CODE PUSH (GitHub/GitLab)
   |
   v
2. LINT + TYPE CHECK
   - ESLint + Prettier
   - TypeScript compiler (tsc --noEmit)
   |
   v
3. UNIT TESTS
   - Jest/Vitest (mock-first, London School TDD)
   - Coverage threshold: 80%
   |
   v
4. INTEGRATION TESTS
   - Docker Compose test environment
   - Test against real PostgreSQL, Redis, Meilisearch
   - API contract tests
   |
   v
5. SECURITY SCAN
   - npm audit / Snyk
   - Container image scan (Trivy)
   - SAST (Semgrep or SonarQube)
   |
   v
6. BUILD DOCKER IMAGES
   - Multi-stage Dockerfile (build -> production)
   - Tag: git SHA + semver
   - Push to container registry
   |
   v
7. DEPLOY TO STAGING
   - Kubernetes apply (Helm chart or Kustomize)
   - Run smoke tests
   - Run E2E tests (Playwright)
   |
   v
8. DEPLOY TO PRODUCTION
   - Rolling update (zero downtime)
   - Canary deployment (10% -> 50% -> 100%)
   - Automatic rollback on error rate > 1%
```

---

## 10. Technology Stack Recommendations

### 10.1 Comparison Matrix

```
+-------------------------------------------------------------------+
| COMPONENT             | OPTION A              | OPTION B           |
|                       | (Node.js/TypeScript)  | (Python/Django)    |
|-----------------------+-----------------------+--------------------|
| API Framework         | Fastify or NestJS     | Django REST / Fast |
|                       |                       | API                |
| Pros                  | Fastest JSON parsing, | Batteries included |
|                       | TypeScript type safe, | Django ORM is      |
|                       | async I/O native,     | excellent, rich    |
|                       | single language FE+BE | ecosystem          |
| Cons                  | Less mature ORM,      | GIL limits CPU     |
|                       | callback complexity   | concurrency, 2     |
|                       |                       | languages if React |
|-----------------------+-----------------------+--------------------|
| OCR / Text Extraction | Tesseract.js (WASM)   | pytesseract +      |
|                       | or child_process to   | pdf2image +        |
|                       | Tesseract binary      | python-docx        |
| RECOMMENDATION        |       PYTHON WINS     |                    |
|                       | Python has far better OCR/NLP ecosystem    |
|-----------------------+-----------------------+--------------------|
| Vector/AI Pipeline    | LangChain.js          | LangChain (Python) |
|                       |                       | or LlamaIndex      |
| RECOMMENDATION        |       PYTHON WINS     |                    |
|                       | More mature, more models, better tooling   |
|-----------------------+-----------------------+--------------------|
| Search Integration    | meilisearch (JS SDK)  | meilisearch        |
|                       | qdrant-js             | (Python SDK)       |
| RECOMMENDATION        |       TIE             |                    |
|                       | Both have excellent SDK support             |
|-----------------------+-----------------------+--------------------|
| Message Queue         | BullMQ (Redis-based)  | Celery (Redis/     |
|                       |                       | RabbitMQ)          |
| RECOMMENDATION        |       TIE             |                    |
|                       | Both mature and production-proven           |
|-----------------------+-----------------------+--------------------|
| Workflow Engine       | Custom state machine  | Django-viewflow    |
|                       | or xstate             | or custom          |
| RECOMMENDATION        |   NODE WINS (xstate)  |                    |
|                       | xstate is excellent for complex state mgmt  |
|-----------------------+-----------------------+--------------------|
| Web UI                | React/Next.js         | Django templates   |
|                       |                       | or separate React  |
| RECOMMENDATION        |    NODE WINS          |                    |
|                       | React ecosystem is far richer for doc UIs   |
+-------------------------------------------------------------------+
```

### 10.2 Recommended Hybrid Architecture

```
RECOMMENDATION: Polyglot architecture with Node.js primary + Python workers

RATIONALE:
  - Node.js/TypeScript for all API services (fast, type-safe, async)
  - Python for OCR workers and RAG/AI pipeline (better ML ecosystem)
  - Both communicate via message queue (language-agnostic boundary)

+-------------------------------------------------------------------+
|                RECOMMENDED TECHNOLOGY STACK                       |
+-------------------------------------------------------------------+

LAYER            | TECHNOLOGY                | WHY
-----------------|---------------------------|---------------------------
API Gateway      | Traefik v3                | Native Docker/K8s, fast,
                 |                           | automatic TLS
                 |                           |
API Services     | NestJS (TypeScript)       | Modular, DI, TypeORM,
                 |                           | OpenAPI auto-gen,
                 |                           | enterprise-grade
                 |                           |
OCR Workers      | Python 3.12 +             | Best OCR ecosystem,
                 | pytesseract + pdf2image   | Tesseract bindings mature,
                 | + python-docx             | Office doc parsing
                 |                           |
RAG/AI Pipeline  | Python 3.12 +             | LangChain most mature,
                 | LangChain + OpenAI SDK    | best model support,
                 |                           | rich chunking tools
                 |                           |
Embedding Model  | OpenAI text-embedding-    | Best quality/cost ratio
                 | 3-small (cloud)           | $0.02/1M tokens
                 | OR all-MiniLM-L6-v2       | Free, runs locally
                 | (local/offline)           |
                 |                           |
LLM              | OpenAI GPT-4o (cloud)     | Best for RAG accuracy
                 | OR Llama 3 70B (local)    | Free, needs GPU
                 |                           |
Relational DB    | PostgreSQL 16             | ACIDit, JSON support,
                 |                           | full-text (backup),
                 |                           | mature replication
                 |                           |
Vector DB        | Qdrant                    | Purpose-built, fast HNSW,
                 |                           | payload filtering (ACL),
                 |                           | easy clustering
                 |                           |
Full-Text Search | Meilisearch               | Typo-tolerant, fast,
                 |                           | facets, easy setup.
                 |                           | Alternative: Typesense
                 |                           |
Message Queue    | BullMQ (on Redis)         | Simple, TypeScript native,
                 |                           | dashboards (Bull Board),
                 |                           | delayed/repeatable jobs
                 |                           |
Cache            | Redis 7                   | Versatile: cache + queue
                 |                           | + pub/sub + sessions
                 |                           |
Object Storage   | MinIO (self-hosted)       | S3-compatible, easy to
                 | or AWS S3 (cloud)         | run locally, cluster mode
                 |                           |
Audit/Analytics  | ClickHouse                | Columnar, fast analytics,
                 |                           | great compression,
                 |                           | append-only fits audit
                 |                           |
Frontend         | Next.js 14 + React 18     | SSR, file preview,
                 | + Tailwind CSS            | document viewer (PDF.js)
                 | + Zustand (state)         |
                 |                           |
Document Viewer  | PDF.js + react-pdf        | In-browser PDF rendering,
                 |                           | text layer for search
                 |                           | highlight overlay
                 |                           |
Monitoring       | Prometheus + Grafana      | Industry standard,
                 | + Loki (logs)             | free, excellent K8s
                 |                           | integration
                 |                           |
Container        | Docker + Docker Compose   | Dev: compose
Orchestration    | + Kubernetes (prod)       | Prod: K8s with Helm
+-------------------------------------------------------------------+
```

### 10.3 Project Directory Structure

```
edms/
  src/
    services/
      auth/
        src/
          controllers/
          middleware/
          models/
          routes/
          services/
          index.ts
        Dockerfile
        package.json
      document/
        src/
          controllers/
          models/
          services/
          storage/
          index.ts
        Dockerfile
        package.json
      search/
        src/
          controllers/
          services/
            keyword-search.service.ts
            semantic-search.service.ts
            hybrid-search.service.ts
          fusion/
            rrf.ts
          index.ts
        Dockerfile
        package.json
      workflow/
        src/
          controllers/
          models/
          state-machine/
          services/
          index.ts
        Dockerfile
        package.json
      rag/
        src/
          controllers/
          services/
          index.ts
        Dockerfile
        package.json
      notification/
        src/
          channels/
          templates/
          services/
          index.ts
        Dockerfile
        package.json
    workers/
      ocr/                    # Python
        ocr_worker/
          __init__.py
          processor.py
          tesseract_wrapper.py
          pdf_extractor.py
          line_segmenter.py
        Dockerfile
        requirements.txt
      indexing/
        src/
          meilisearch-indexer.ts
          qdrant-indexer.ts
          embedding-client.ts
          index.ts
        Dockerfile
        package.json
      embedding/               # Python
        embedding_worker/
          __init__.py
          embedder.py
          chunker.py
        Dockerfile
        requirements.txt
    shared/
      types/                   # Shared TypeScript types
        document.ts
        search.ts
        auth.ts
        workflow.ts
      utils/
        logger.ts
        validation.ts
      constants/
        permissions.ts
        mime-types.ts
    frontend/
      app/                     # Next.js app directory
        (auth)/
        (dashboard)/
        documents/
        search/
        workflows/
        admin/
      components/
        document-viewer/
        search-results/
        workflow-board/
      lib/
        api-client.ts
        auth.ts
  config/
    traefik/
    nginx/
    prometheus/
    grafana/
  scripts/
    init.sql
    seed.ts
    migrate.ts
  tests/
    integration/
    e2e/
  docker-compose.yml
  docker-compose.prod.yml
  package.json
  tsconfig.json
```

---

## Appendix A: Where to Fill Gaps

```
IMPLEMENTATION PRIORITY ORDER:

PHASE 1 - FOUNDATION (Weeks 1-3)
  [x] PostgreSQL schema (Section 4.5)
  [ ] Auth Service with JWT + RBAC
  [ ] Document upload + S3 storage
  [ ] Basic API Gateway (Traefik)
  [ ] Docker Compose dev environment

PHASE 2 - OCR PIPELINE (Weeks 4-5)
  [ ] OCR worker (Python + Tesseract)
  [ ] Page and line extraction with bounding boxes
  [ ] BullMQ queue integration
  [ ] Document status tracking (processing -> ready)

PHASE 3 - SEARCH (Weeks 6-8)
  [ ] Meilisearch indexing (pages + lines)
  [ ] Keyword search with ACL filtering
  [ ] Qdrant setup + embedding pipeline
  [ ] Semantic search
  [ ] Hybrid search with RRF fusion
  [ ] Smart search (page+line precision results)

PHASE 4 - RAG/AI (Weeks 9-10)
  [ ] RAG pipeline with citation mapping
  [ ] Conversation history
  [ ] LLM integration (OpenAI or local)

PHASE 5 - WORKFLOWS + ACL (Weeks 11-13)
  [ ] Workflow engine (state machine)
  [ ] Object-level ACL
  [ ] Notification service
  [ ] Audit logging (ClickHouse)

PHASE 6 - FRONTEND (Weeks 14-17)
  [ ] Next.js app with authentication
  [ ] Document viewer with PDF.js
  [ ] Search UI with highlighting
  [ ] RAG chat interface
  [ ] Workflow board
  [ ] Admin panel

PHASE 7 - PRODUCTION (Weeks 18-20)
  [ ] Kubernetes manifests / Helm chart
  [ ] CI/CD pipeline
  [ ] Monitoring (Prometheus + Grafana)
  [ ] Load testing
  [ ] Security hardening
```

---

## Appendix B: Key Architectural Decisions

```
ADR-001: Polyglot Architecture (Node.js + Python)
  Status:   Accepted
  Context:  Need best-in-class OCR and ML tooling alongside fast API servers
  Decision: Node.js/TypeScript for API services, Python for ML workers
  Reason:   Python's ML/OCR ecosystem is unmatched; Node.js excels at I/O

ADR-002: Meilisearch over Elasticsearch
  Status:   Accepted
  Context:  Need full-text search with typo tolerance
  Decision: Use Meilisearch instead of Elasticsearch
  Reason:   10x simpler to operate, faster for datasets under 10M docs,
            built-in typo tolerance, lower resource usage

ADR-003: Qdrant over Pinecone/Weaviate
  Status:   Accepted
  Context:  Need vector database for semantic search
  Decision: Use Qdrant
  Reason:   Self-hostable, payload filtering (critical for ACL),
            excellent performance, Rust-based (low resource usage)

ADR-004: Page-Level Embedding Granularity
  Status:   Accepted
  Context:  What unit to embed for semantic search?
  Decision: Page-level chunks (not paragraph or sentence)
  Reason:   Pages are natural document boundaries, users reference pages,
            manageable vector count, sufficient context per chunk

ADR-005: Pre-computed ACL in Search Indexes
  Status:   Accepted
  Context:  How to enforce ACL during search queries?
  Decision: Store acl_read arrays in Meilisearch and Qdrant
  Trade-off: ACL changes require re-indexing affected documents
  Reason:   Search-time ACL joins are too slow; ACL changes are infrequent

ADR-006: Async Processing Pipeline
  Status:   Accepted
  Context:  OCR and embedding are CPU/GPU intensive
  Decision: Queue-based async processing with BullMQ
  Reason:   Non-blocking uploads, independent scaling of workers,
            retry handling, progress tracking

ADR-007: Hybrid Search with RRF Fusion
  Status:   Accepted
  Context:  Keyword search misses semantics; vector search misses exact terms
  Decision: Run both in parallel, fuse with Reciprocal Rank Fusion
  Reason:   RRF is simple, parameter-free, and outperforms either alone
```
