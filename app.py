"""Smart Search Tool: Source-First PDF search with highlighted screenshots."""

import os
import json
import httpx
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response, JSONResponse

from pdf_processor import PDFProcessor
from search_engine import SearchEngine

app = FastAPI(title="Smart PDF Search")
pdf_processor = PDFProcessor(cache_dir="cache")
search_engine = SearchEngine()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Track the currently loaded PDF
current_pdf: dict = {"path": None, "name": None, "page_count": 0}

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text()


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and index it for searching."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    pdf_path = UPLOAD_DIR / file.filename
    content = await file.read()
    pdf_path.write_bytes(content)

    # Extract and index
    pages = pdf_processor.extract_pages(str(pdf_path))
    search_engine.index(pages)
    page_count = pdf_processor.get_page_count(str(pdf_path))

    current_pdf["path"] = str(pdf_path)
    current_pdf["name"] = file.filename
    current_pdf["page_count"] = page_count

    return {
        "filename": file.filename,
        "page_count": page_count,
        "indexed_pages": len(pages),
        "total_lines": sum(len(p.lines) for p in pages)
    }


@app.get("/api/page/{page_number}")
async def get_page_image(page_number: int):
    """Get a clean page image (no highlights)."""
    if not current_pdf["path"]:
        raise HTTPException(400, "No PDF uploaded")
    if page_number < 1 or page_number > current_pdf["page_count"]:
        raise HTTPException(400, f"Page must be between 1 and {current_pdf['page_count']}")

    img = pdf_processor.get_page_image(current_pdf["path"], page_number)
    return Response(content=img, media_type="image/png")


@app.post("/api/search")
async def search(body: dict):
    """Search the PDF and return results with highlighted page screenshots."""
    query = body.get("query", "").strip()
    if not query:
        raise HTTPException(400, "Query is required")
    if not current_pdf["path"]:
        raise HTTPException(400, "No PDF uploaded")

    results = search_engine.search(query, top_k=5)

    response_results = []
    for result in results:
        # Collect texts to highlight on this page
        highlight_texts = [
            ml["text"] for ml in result.matched_lines[:5]
        ]

        # Build matched lines response
        matched_lines = []
        highlight_bboxes = []
        for ml in result.matched_lines:
            matched_lines.append({
                "line_number": ml["line_number"],
                "text": ml["text"],
                "score": round(ml["score"], 3),
                "match_type": ml.get("match_type", "semantic"),
                "bbox": ml["bbox"],
                "context_before": ml.get("context_before", []),
                "context_after": ml.get("context_after", []),
            })
            highlight_bboxes.append(ml["bbox"])

        response_results.append({
            "page_number": result.page_number,
            "page_score": round(result.page_score, 3),
            "matched_lines": matched_lines,
            "highlight_image_url": f"/api/page/{result.page_number}/highlighted",
            "clean_image_url": f"/api/page/{result.page_number}",
        })

    # Build RAG context: full page text from top matching pages
    rag_context = []
    for result in results[:3]:
        page_text = search_engine.get_page_text(result.page_number)
        if page_text:
            rag_context.append({
                "page_number": result.page_number,
                "text": page_text[:2000],  # cap per page to stay within token limits
            })

    # Get AI answer via RAG
    ai_summary = None
    ai_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if ai_key and rag_context:
        ai_summary = await _generate_rag_answer(query, rag_context, ai_key)

    return {
        "query": query,
        "topic": search_engine.extract_topic(query),
        "total_results": len(response_results),
        "results": response_results,
        "ai_summary": ai_summary,
        "pdf_name": current_pdf["name"],
    }


@app.get("/api/page/{page_number}/highlighted")
async def get_highlighted_page(page_number: int, texts: str = ""):
    """Get a page image with specific texts highlighted."""
    if not current_pdf["path"]:
        raise HTTPException(400, "No PDF uploaded")

    highlight_texts = json.loads(texts) if texts else []

    if highlight_texts:
        img = pdf_processor.get_highlighted_page_image(
            current_pdf["path"], page_number, highlight_texts
        )
    else:
        img = pdf_processor.get_page_image(current_pdf["path"], page_number)

    return Response(content=img, media_type="image/png")


@app.post("/api/page/{page_number}/highlighted")
async def get_highlighted_page_post(page_number: int, body: dict):
    """Get a page image with specific texts highlighted (POST for long text lists)."""
    if not current_pdf["path"]:
        raise HTTPException(400, "No PDF uploaded")

    highlight_texts = body.get("texts", [])

    if highlight_texts:
        img = pdf_processor.get_highlighted_page_image(
            current_pdf["path"], page_number, highlight_texts
        )
    else:
        img = pdf_processor.get_page_image(current_pdf["path"], page_number)

    return Response(content=img, media_type="image/png")


async def _generate_rag_answer(query: str, rag_context: list[dict], api_key: str) -> dict:
    """RAG: Send full page context to LLM to understand intent and generate answer."""
    # Build rich context from full page text
    sources_text = ""
    for ctx in rag_context:
        sources_text += f"\n\n--- PAGE {ctx['page_number']} ---\n{ctx['text']}"

    if not sources_text.strip():
        return None

    system_prompt = (
        "You are a document analysis assistant. The user will ask a question about a PDF document. "
        "You are given the full text of the most relevant pages from that document. "
        "Answer the question based ONLY on the provided page content. "
        "Be concise but thorough. Use [Page X] citations to reference which page your answer comes from. "
        "If the answer spans multiple pages, cite each page. "
        "If the document doesn't contain enough info to answer, say so clearly."
    )

    user_prompt = f"DOCUMENT PAGES:{sources_text}\n\nQUESTION: {query}"

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    try:
        if anthropic_key:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 500,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}]
                    }
                )
                data = resp.json()
                text = data.get("content", [{}])[0].get("text", "No response")
            model_name = "Claude Haiku 4.5"

        elif openai_key:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openai_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_tokens": 500
                    }
                )
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
            model_name = "GPT-4o Mini"
        else:
            return None

        return {
            "text": text,
            "disclaimer": f"AI answer by {model_name}. Verify against the highlighted source in the document.",
            "enabled": True,
            "model": model_name
        }

    except Exception as e:
        return {
            "text": f"AI answer unavailable: {str(e)}",
            "disclaimer": "Could not generate AI answer.",
            "enabled": False
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
