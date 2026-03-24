"""Standalone test UI for the RAG chunking pipeline.

No Docker, no PostgreSQL — runs entirely in-memory.
Upload a PDF, ask questions, verify the pipeline works.

Run:
    cd /Users/saiteja/Documents/Dev/EDMS/src/smart-search
    .venv/bin/python tools/test_ui.py

Then open http://localhost:8888
"""

import sys
import os
import re
import json
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env_file = Path(__file__).resolve().parent.parent.parent.parent / "docker" / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

import numpy as np
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from fastembed import TextEmbedding
from pymupdf4llm_parser import PyMuPDF4LLMParser
from chunking_pipeline import ChunkingPipeline

# ---------------------------------------------------------------------------
# In-memory store (no PostgreSQL needed)
# ---------------------------------------------------------------------------

class InMemoryStore:
    def __init__(self):
        self.documents: dict[str, dict] = {}  # doc_key -> {name, chunks, embeddings}
        self.model = None

    def _load_model(self):
        if self.model is None:
            print("Loading embedding model...")
            self.model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
            print("Model loaded.")

    def index(self, doc_key: str, doc_name: str, chunks: list):
        self._load_model()
        children = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
        parents = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]

        # Embed children using enriched text
        child_texts = [c.metadata.get("enriched_text", c.text) for c in children]
        embeddings = list(self.model.embed(child_texts))

        self.documents[doc_key] = {
            "name": doc_name,
            "children": children,
            "parents": parents,
            "embeddings": np.array(embeddings),
            "child_texts": child_texts,
        }
        print(f"Indexed {doc_key}: {len(parents)}P + {len(children)}C")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self._load_model()
        query_emb = np.array(list(self.model.embed([query]))[0])

        all_results = []
        for doc_key, doc in self.documents.items():
            if len(doc["children"]) == 0:
                continue
            # Cosine similarity
            sims = doc["embeddings"] @ query_emb / (
                np.linalg.norm(doc["embeddings"], axis=1) * np.linalg.norm(query_emb) + 1e-10
            )
            for idx in np.argsort(sims)[::-1][:top_k]:
                child = doc["children"][int(idx)]
                all_results.append({
                    "score": float(sims[int(idx)]),
                    "text": child.text,
                    "section": child.metadata.get("section", ""),
                    "page": child.metadata.get("page_number", 0),
                    "doc_name": doc["name"],
                    "doc_key": doc_key,
                    "parent_id": child.metadata.get("parent_id", ""),
                })

        all_results.sort(key=lambda r: r["score"], reverse=True)
        return all_results[:top_k]

    def get_parent(self, doc_key: str, parent_id: str) -> str | None:
        doc = self.documents.get(doc_key)
        if not doc:
            return None
        for p in doc["parents"]:
            if p.metadata.get("parent_id") == parent_id:
                return p.text
        return None

    def list_documents(self) -> list[dict]:
        return [
            {"key": k, "name": v["name"], "chunks": len(v["children"]) + len(v["parents"])}
            for k, v in self.documents.items()
        ]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

parser = PyMuPDF4LLMParser()
pipeline = ChunkingPipeline()
store = InMemoryStore()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files")

    pdf_path = UPLOAD_DIR / file.filename
    pdf_path.write_bytes(await file.read())

    structure = parser.extract_structure(str(pdf_path))
    doc_key = re.sub(r"[^a-z0-9]+", "-", file.filename.lower()).strip("-")
    chunks = pipeline.chunk_document(structure, file.filename, document_key=doc_key)
    store.index(doc_key, file.filename, chunks)

    parents = sum(1 for c in chunks if c.metadata.get("chunk_type") == "parent")
    children = len(chunks) - parents

    return {
        "filename": file.filename,
        "doc_type": structure.doc_type,
        "title": structure.title,
        "pages": len(structure.pages),
        "parents": parents,
        "children": children,
        "total": len(chunks),
    }


@app.post("/api/chat")
async def chat(body: dict):
    query = body.get("query", "").strip()
    if not query:
        raise HTTPException(400, "Query required")

    results = store.search(query, top_k=5)
    if not results:
        return {"answer": "No documents indexed yet. Please upload a PDF first.", "sources": []}

    # Build RAG context from parents
    rag_parts = []
    seen_parents = set()
    for r in results[:3]:
        pid = r["parent_id"]
        if pid in seen_parents:
            continue
        seen_parents.add(pid)
        parent_text = store.get_parent(r["doc_key"], pid)
        context = parent_text if parent_text else r["text"]
        # Trim to 4000 chars
        if len(context) > 4000:
            context = context[:4000] + "..."
        rag_parts.append(f"--- {r['doc_name']}, Page {r['page']}, Section: {r['section']} ---\n{context}")

    sources_text = "\n\n".join(rag_parts)

    # Call Groq
    api_key = os.environ.get("GROQ_API_KEY")
    answer = None
    if api_key:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": (
                                "You are a document analysis assistant. Answer based ONLY on the provided excerpts. "
                                "Cite sources as [Page X]. If the excerpts don't contain enough info, say so."
                            )},
                            {"role": "user", "content": f"DOCUMENT EXCERPTS:\n{sources_text}\n\nQUESTION: {query}"},
                        ],
                        "max_tokens": 500,
                        "temperature": 0,
                    },
                )
                if resp.status_code == 200:
                    answer = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            answer = f"LLM error: {e}"

    if not answer:
        answer = "Groq API key not set. Here are the top matching chunks:\n\n" + "\n\n".join(
            f"[Page {r['page']}] {r['section']}: {r['text'][:200]}..." for r in results[:3]
        )

    sources = [
        {"page": r["page"], "section": r["section"], "score": round(r["score"], 3),
         "doc_name": r["doc_name"], "preview": r["text"][:150]}
        for r in results[:5]
    ]

    return {"answer": answer, "sources": sources}


@app.get("/api/documents")
async def documents():
    return {"documents": store.list_documents()}


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Pipeline Tester</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }

.header { background: #1e293b; border-bottom: 1px solid #334155; padding: 14px 24px; display: flex; align-items: center; gap: 16px; }
.header h1 { font-size: 18px; color: #f8fafc; }
.header .badge { background: #3b82f620; color: #60a5fa; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; }

.main { display: flex; flex: 1; overflow: hidden; }

/* Sidebar */
.sidebar { width: 300px; background: #1e293b; border-right: 1px solid #334155; display: flex; flex-direction: column; padding: 16px; gap: 12px; }
.upload-zone { border: 2px dashed #334155; border-radius: 12px; padding: 24px; text-align: center; cursor: pointer; transition: border-color 0.2s; }
.upload-zone:hover { border-color: #60a5fa; }
.upload-zone.dragover { border-color: #60a5fa; background: #3b82f610; }
.upload-zone input { display: none; }
.upload-zone .icon { font-size: 32px; margin-bottom: 8px; }
.upload-zone p { color: #64748b; font-size: 13px; }
.upload-status { font-size: 12px; color: #10b981; padding: 8px; background: #10b98110; border-radius: 8px; display: none; }
.doc-list { flex: 1; overflow-y: auto; }
.doc-item { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; margin-bottom: 6px; font-size: 12px; }
.doc-item .name { color: #f8fafc; font-weight: 600; margin-bottom: 2px; }
.doc-item .meta { color: #64748b; }

/* Chat */
.chat-area { flex: 1; display: flex; flex-direction: column; }
.messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
.msg { max-width: 80%; padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.6; }
.msg.user { align-self: flex-end; background: #3b82f6; color: white; border-bottom-right-radius: 4px; }
.msg.bot { align-self: flex-start; background: #1e293b; border: 1px solid #334155; border-bottom-left-radius: 4px; }
.msg.bot .answer { white-space: pre-wrap; }
.sources { margin-top: 10px; padding-top: 10px; border-top: 1px solid #334155; }
.sources .title { font-size: 11px; color: #64748b; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; }
.source-card { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 8px 10px; margin-bottom: 4px; font-size: 12px; }
.source-card .sec { color: #60a5fa; font-weight: 600; }
.source-card .preview { color: #94a3b8; margin-top: 2px; }
.source-card .score { color: #10b981; font-size: 11px; float: right; }

.input-bar { padding: 16px 20px; background: #1e293b; border-top: 1px solid #334155; display: flex; gap: 10px; }
.input-bar input { flex: 1; background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px 16px; color: #f8fafc; font-size: 14px; outline: none; }
.input-bar input:focus { border-color: #3b82f6; }
.input-bar button { background: #3b82f6; color: white; border: none; border-radius: 8px; padding: 12px 20px; font-size: 14px; font-weight: 600; cursor: pointer; }
.input-bar button:hover { background: #2563eb; }
.input-bar button:disabled { background: #334155; cursor: not-allowed; }

.loading { display: inline-block; width: 16px; height: 16px; border: 2px solid #64748b; border-top-color: #60a5fa; border-radius: 50%; animation: spin 0.6s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

.welcome { text-align: center; color: #475569; margin: auto; }
.welcome h2 { font-size: 20px; color: #64748b; margin-bottom: 8px; }
.welcome p { font-size: 14px; }
</style>
</head>
<body>

<div class="header">
    <h1>RAG Pipeline Tester</h1>
    <span class="badge">Stages 1-6 Active</span>
</div>

<div class="main">
    <div class="sidebar">
        <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" accept=".pdf" />
            <div class="icon">+</div>
            <p>Upload PDF to index</p>
        </div>
        <div class="upload-status" id="uploadStatus"></div>
        <div class="doc-list" id="docList"></div>
    </div>

    <div class="chat-area">
        <div class="messages" id="messages">
            <div class="welcome">
                <h2>Upload a PDF and ask questions</h2>
                <p>Tests the full pipeline: extraction, doc type detection, skip rules, parent-child chunking, enrichment, and search</p>
            </div>
        </div>
        <div class="input-bar">
            <input type="text" id="queryInput" placeholder="Ask a question about your document..." />
            <button id="sendBtn" onclick="sendQuery()">Send</button>
        </div>
    </div>
</div>

<script>
const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');
const uploadStatus = document.getElementById('uploadStatus');
const docList = document.getElementById('docList');
const messages = document.getElementById('messages');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');

// Drag and drop
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => { if (fileInput.files[0]) uploadFile(fileInput.files[0]); });
queryInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendQuery(); });

async function uploadFile(file) {
    uploadStatus.style.display = 'block';
    uploadStatus.textContent = 'Processing: extracting structure, detecting doc type, chunking...';
    uploadStatus.style.color = '#f59e0b';

    const form = new FormData();
    form.append('file', file);

    try {
        const resp = await fetch('/api/upload', { method: 'POST', body: form });
        const data = await resp.json();
        uploadStatus.style.color = '#10b981';
        uploadStatus.textContent = `Indexed: ${data.total} chunks (${data.parents}P + ${data.children}C) | Type: ${data.doc_type}`;
        refreshDocs();
    } catch (e) {
        uploadStatus.style.color = '#ef4444';
        uploadStatus.textContent = 'Upload failed: ' + e.message;
    }
}

async function refreshDocs() {
    const resp = await fetch('/api/documents');
    const data = await resp.json();
    docList.innerHTML = data.documents.map(d =>
        '<div class="doc-item"><div class="name">' + d.name + '</div><div class="meta">' + d.chunks + ' chunks</div></div>'
    ).join('');
}

async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    // Remove welcome
    const welcome = messages.querySelector('.welcome');
    if (welcome) welcome.remove();

    // Add user message
    messages.innerHTML += '<div class="msg user">' + escHtml(query) + '</div>';
    queryInput.value = '';
    sendBtn.disabled = true;

    // Loading
    const loadingId = 'loading-' + Date.now();
    messages.innerHTML += '<div class="msg bot" id="' + loadingId + '"><div class="loading"></div></div>';
    messages.scrollTop = messages.scrollHeight;

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });
        const data = await resp.json();

        let sourcesHtml = '';
        if (data.sources && data.sources.length > 0) {
            sourcesHtml = '<div class="sources"><div class="title">Sources</div>' +
                data.sources.map(s =>
                    '<div class="source-card"><span class="score">' + s.score + '</span>' +
                    '<div class="sec">[Page ' + s.page + '] ' + escHtml(s.section) + '</div>' +
                    '<div class="preview">' + escHtml(s.preview) + '</div></div>'
                ).join('') + '</div>';
        }

        document.getElementById(loadingId).innerHTML =
            '<div class="answer">' + escHtml(data.answer) + '</div>' + sourcesHtml;
    } catch (e) {
        document.getElementById(loadingId).innerHTML = '<div class="answer" style="color:#ef4444">Error: ' + e.message + '</div>';
    }

    sendBtn.disabled = false;
    messages.scrollTop = messages.scrollHeight;
}

function escHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }

refreshDocs();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    print("\n  RAG Pipeline Tester")
    print("  http://localhost:8899\n")
    uvicorn.run(app, host="0.0.0.0", port=8899)
