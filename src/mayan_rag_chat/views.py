"""Views for the RAG Chat app — proxies to the RAG sidecar service."""

import os
import json
import logging

import httpx
from django.apps import apps
from django.http import JsonResponse
from django.template.response import TemplateResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

logger = logging.getLogger(__name__)

RAG_URL = os.environ.get('RAG_SIDECAR_URL', 'http://rag-sidecar:8080')
# Browser-accessible URL for JavaScript calls (not internal Docker hostname)
RAG_PUBLIC_URL = os.environ.get('RAG_SIDECAR_PUBLIC_URL', 'http://localhost:8080')


class ChatView(View):
    """Main chatbot page."""
    def get(self, request):
        stats = _get_stats()
        return TemplateResponse(request, 'rag_chat/chat.html', {
            'title': 'RAG Document Chat',
            'stats': stats,
        })


class ChunkInspectView(View):
    """Inspect stored parent and child chunks for a document."""

    def get(self, request, document_id=None):
        Document = apps.get_model('documents', 'Document')
        document_id = document_id or request.GET.get('document_id')

        if not document_id:
            return TemplateResponse(request, 'rag_chat/chunks.html', {
                'title': 'Chunk Inspector',
                'document': None,
                'document_id': '',
                'stats': _get_stats(),
            })

        try:
            document_id = int(document_id)
        except (TypeError, ValueError):
            return TemplateResponse(request, 'rag_chat/chunks.html', {
                'title': 'Chunk Inspector',
                'document': None,
                'document_id': document_id,
                'error': 'Document ID must be a number.',
                'stats': _get_stats(),
            }, status=400)

        try:
            document = Document.objects.get(pk=document_id)
        except Document.DoesNotExist:
            return TemplateResponse(request, 'rag_chat/chunks.html', {
                'title': 'Chunk Inspector',
                'document': None,
                'document_id': document_id,
                'error': f'Document {document_id} was not found in Mayan.',
                'stats': _get_stats(),
            }, status=404)

        try:
            resp = httpx.get(
                f'{RAG_URL}/api/mayan/chunks/{document_id}',
                params={'limit': 200},
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError as exc:
            error = f'Chunk data is not available yet for document {document_id}.'
            if exc.response.status_code != 404:
                error = f'Failed to load chunk data: {exc.response.text[:200]}'
            return TemplateResponse(request, 'rag_chat/chunks.html', {
                'title': 'Chunk Inspector',
                'document': document,
                'document_id': document_id,
                'error': error,
                'stats': _get_stats(),
            }, status=exc.response.status_code)
        except Exception as exc:
            return TemplateResponse(request, 'rag_chat/chunks.html', {
                'title': 'Chunk Inspector',
                'document': document,
                'document_id': document_id,
                'error': f'Failed to connect to the RAG sidecar: {exc}',
                'stats': _get_stats(),
            }, status=502)

        summary = payload.get('summary', {})
        parents = payload.get('parents', [])
        children = payload.get('children', [])

        return TemplateResponse(request, 'rag_chat/chunks.html', {
            'title': f'Chunk Inspector: {document.label}',
            'document': document,
            'document_id': document_id,
            'summary': summary,
            'parents': parents,
            'children': children,
            'stats': _get_stats(),
        })


@method_decorator(csrf_exempt, name='dispatch')
class IndexAllView(View):
    """Index all documents by syncing each one to the RAG sidecar."""

    def get(self, request):
        Document = apps.get_model('documents', 'Document')
        stats = _get_stats()
        return TemplateResponse(request, 'rag_chat/index_all.html', {
            'title': 'Index All Documents for RAG',
            'document_count': Document.objects.count(),
            'stats': stats,
        })

    def post(self, request):
        Document = apps.get_model('documents', 'Document')
        indexed = 0
        errors = []

        for doc in Document.objects.all():
            try:
                resp = httpx.post(
                    f'{RAG_URL}/api/mayan/sync/{doc.pk}',
                    timeout=300,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    indexed += data.get('total_chunks', 0)
                else:
                    errors.append({"doc_id": doc.pk, "error": resp.text[:200]})
            except Exception as e:
                errors.append({"doc_id": doc.pk, "error": str(e)})

        return JsonResponse({
            "status": "ok",
            "chunks_indexed": indexed,
            "documents": Document.objects.count(),
            "errors": errors,
            "stats": _get_stats(),
        })


@method_decorator(csrf_exempt, name='dispatch')
class SearchAPIView(View):
    """Proxy search queries to the RAG sidecar."""

    def post(self, request):
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        query = body.get("query", "").strip()
        if not query:
            return JsonResponse({"error": "Query required"}, status=400)

        try:
            resp = httpx.post(
                f'{RAG_URL}/api/search',
                json={
                    "query": query,
                    "document_id": body.get("document_id"),
                },
                timeout=60,
            )
            data = resp.json()

            # Transform results — include bounding boxes for PDF highlighting
            results = []
            for r in data.get("results", []):
                results.append({
                    "page_number": r.get("page_number", 0),
                    "page_score": r.get("page_score", 0),
                    "label": r.get("document_name") or data.get("pdf_name", ""),
                    "doc_id": r.get("document_id"),
                    "highlight_bboxes": r.get("highlight_bboxes", []),
                    "chunks": r.get("chunks", []),
                })

            return JsonResponse({
                "query": query,
                "results": results[:5],
                "ai_answer": data.get("ai_summary"),
                "stats": _get_stats(),
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=502)


@method_decorator(csrf_exempt, name='dispatch')
class IndexDocumentView(View):
    """Index a single document."""
    def post(self, request, document_id):
        try:
            resp = httpx.post(
                f'{RAG_URL}/api/mayan/sync/{document_id}',
                timeout=300,
            )
            return JsonResponse(resp.json())
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=502)


class NudgeView(View):
    """AI Insights: show related documents for a given document."""

    def get(self, request, document_id=None):
        Document = apps.get_model('documents', 'Document')
        document = None
        doc_type = ""

        if document_id:
            try:
                document = Document.objects.get(pk=int(document_id))
                doc_type = document.document_type.label if document.document_type else ""
            except Document.DoesNotExist:
                pass

        return TemplateResponse(request, 'rag_chat/nudge.html', {
            'title': f'AI Insights: {document.label}' if document else 'AI Insights',
            'document': document,
            'doc_type': doc_type,
            'rag_url': RAG_PUBLIC_URL,
            'stats': _get_stats(),
        })


class SummarizeView(View):
    """Executive Briefing: batch summarize a collection of documents."""

    def get(self, request):
        return TemplateResponse(request, 'rag_chat/summarize.html', {
            'title': 'Executive Briefing',
            'rag_url': RAG_PUBLIC_URL,
            'stats': _get_stats(),
        })


def _get_stats():
    """Get stats from the RAG sidecar."""
    try:
        # Quick check — if sidecar is up, get its status
        resp = httpx.get(f'{RAG_URL}/api/mayan/status', timeout=5)
        if resp.status_code == 200:
            return {"total_chunks": 0, "unique_docs": 0, "parents": 0, "children": 0, "sidecar": "connected"}
    except Exception:
        pass
    return {"total_chunks": 0, "unique_docs": 0, "parents": 0, "children": 0, "sidecar": "disconnected"}
