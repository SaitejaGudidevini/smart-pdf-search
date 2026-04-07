"""Mayan EDMS Bridge: Fetch documents and OCR text from Mayan's REST API."""

import os
import httpx
from pathlib import Path
from dataclasses import dataclass


@dataclass
class MayanDocument:
    """A document fetched from Mayan EDMS."""
    id: int
    label: str
    doc_type: str
    page_count: int
    version_id: int | None
    file_path: str  # local path after download
    metadata: dict


class MayanBridge:
    """Connects to Mayan EDMS via REST API to fetch documents and OCR text."""

    def __init__(self, base_url: str = None, username: str = None, password: str = None):
        self.base_url = (base_url or os.environ.get("MAYAN_API_URL", "http://localhost:8000/api/v4")).rstrip("/")
        self.username = username or os.environ.get("MAYAN_USERNAME", "admin")
        self.password = password or os.environ.get("MAYAN_PASSWORD", "admin123")
        self._token = None
        self.download_dir = Path("uploads")
        self.download_dir.mkdir(exist_ok=True)

    async def _get_token(self) -> str:
        """Authenticate with Mayan and get auth token."""
        if self._token:
            return self._token
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{self.base_url}/auth/token/obtain/?format=json",
                json={"username": self.username, "password": self.password},
            )
            resp.raise_for_status()
            self._token = resp.json()["token"]
            return self._token

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Token {self._token}"}

    async def get_document_cabinet_path(self, doc_id: int) -> dict | None:
        """Get the full cabinet hierarchy for a document.

        Returns a landmark dict that travels with every chunk:
        {
            "cabinet_id": 8,
            "cabinet_label": "Q2 2025 Financial Filings",
            "cabinet_path": "Apple Inc. / Q2 2025 Financial Filings",
            "company_id": 3,
            "company_name": "Apple Inc.",
            "hierarchy": [
                {"id": 3, "label": "Apple Inc.", "level": 0},
                {"id": 8, "label": "Q2 2025 Financial Filings", "level": 1},
            ]
        }
        """
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.base_url}/documents/{doc_id}/cabinets/?format=json",
                headers=self._auth_headers(),
            )
            if resp.status_code != 200:
                return None

            cabs = resp.json().get("results", [])
            if not cabs:
                return None

            # Use the first cabinet (document's primary location)
            cab = cabs[0]
            full_path = cab.get("full_path", cab.get("label", ""))
            path_parts = [p.strip() for p in full_path.split("/") if p.strip()]

            # Walk up the hierarchy to build landmark
            hierarchy = []
            current_id = cab["id"]
            current_label = cab.get("label", "")
            parent_id = cab.get("parent_id")

            # Build from leaf to root
            chain = [{"id": current_id, "label": current_label}]
            while parent_id:
                parent_resp = await client.get(
                    f"{self.base_url}/cabinets/{parent_id}/?format=json",
                    headers=self._auth_headers(),
                )
                if parent_resp.status_code != 200:
                    break
                parent = parent_resp.json()
                chain.append({"id": parent["id"], "label": parent["label"]})
                parent_id = parent.get("parent_id")

            # Reverse: root first
            chain.reverse()
            for i, node in enumerate(chain):
                node["level"] = i
            hierarchy = chain

            return {
                "cabinet_id": cab["id"],
                "cabinet_label": cab.get("label", ""),
                "cabinet_path": full_path,
                "company_id": hierarchy[0]["id"] if hierarchy else None,
                "company_name": hierarchy[0]["label"] if hierarchy else None,
                "hierarchy": hierarchy,
            }

    async def get_cabinet_document_ids(self, cabinet_id: int) -> list[int]:
        """Get all document IDs in a cabinet and its sub-cabinets (recursive).

        If cabinet_id is a company-level cabinet, returns docs from all
        contract sub-cabinets under it.
        """
        token = await self._get_token()
        doc_ids = set()

        async with httpx.AsyncClient(timeout=15) as client:
            # Get documents directly in this cabinet
            resp = await client.get(
                f"{self.base_url}/cabinets/{cabinet_id}/documents/?format=json&page_size=100",
                headers=self._auth_headers(),
            )
            if resp.status_code == 200:
                for doc in resp.json().get("results", []):
                    doc_ids.add(doc["id"])

            # Get child cabinets and recurse
            resp = await client.get(
                f"{self.base_url}/cabinets/{cabinet_id}/?format=json",
                headers=self._auth_headers(),
            )
            if resp.status_code == 200:
                children = resp.json().get("children", [])
                for child in children:
                    child_id = child["id"] if isinstance(child, dict) else child
                    # Get docs from child cabinet
                    child_resp = await client.get(
                        f"{self.base_url}/cabinets/{child_id}/documents/?format=json&page_size=100",
                        headers=self._auth_headers(),
                    )
                    if child_resp.status_code == 200:
                        for doc in child_resp.json().get("results", []):
                            doc_ids.add(doc["id"])

        return sorted(doc_ids)

    async def list_cabinets(self) -> list[dict]:
        """List all cabinets with their hierarchy."""
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.base_url}/cabinets/?format=json&page_size=100",
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            return resp.json().get("results", [])

    async def list_documents(self, page: int = 1) -> list[dict]:
        """List documents from Mayan EDMS."""
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.base_url}/documents/?format=json&page={page}",
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])

    async def get_document(self, doc_id: int) -> dict:
        """Get document details by ID."""
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.base_url}/documents/{doc_id}/?format=json",
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            return resp.json()

    async def get_document_file(self, doc_id: int) -> str:
        """Download the latest file of a document. Returns local file path."""
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=60) as client:
            # Get document files
            resp = await client.get(
                f"{self.base_url}/documents/{doc_id}/files/?format=json",
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            files = resp.json().get("results", [])
            if not files:
                raise ValueError(f"No files for document {doc_id}")

            latest = files[0]
            file_id = latest["id"]
            filename = latest.get("filename", f"doc_{doc_id}.pdf")

            # Download file
            resp = await client.get(
                f"{self.base_url}/documents/{doc_id}/files/{file_id}/download/?format=json",
                headers=self._auth_headers(),
                follow_redirects=True,
            )
            resp.raise_for_status()

            file_path = self.download_dir / filename
            file_path.write_bytes(resp.content)
            return str(file_path)

    async def get_ocr_text(self, doc_id: int) -> list[dict]:
        """Get OCR text for all pages of a document.

        Returns: [{"page_number": 1, "text": "..."}, ...]
        """
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=30) as client:
            # Get document versions
            resp = await client.get(
                f"{self.base_url}/documents/{doc_id}/versions/?format=json",
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            versions = resp.json().get("results", [])
            if not versions:
                return []

            version_id = versions[0]["id"]

            # Get pages
            resp = await client.get(
                f"{self.base_url}/documents/{doc_id}/versions/{version_id}/pages/?format=json",
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            pages = resp.json().get("results", [])

            # Get OCR text for each page
            ocr_pages = []
            for i, page in enumerate(pages):
                page_id = page["id"]
                resp = await client.get(
                    f"{self.base_url}/documents/{doc_id}/versions/{version_id}/pages/{page_id}/ocr/?format=json",
                    headers=self._auth_headers(),
                )
                if resp.status_code == 200:
                    ocr_data = resp.json()
                    text = ocr_data.get("content", "") or ocr_data.get("text", "")
                    ocr_pages.append({"page_number": i + 1, "text": text})

            return ocr_pages

    async def get_document_metadata(self, doc_id: int) -> dict:
        """Get metadata (tags, cabinets, custom fields) for a document."""
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=15) as client:
            doc = await self.get_document(doc_id)

            # Get tags
            resp = await client.get(
                f"{self.base_url}/documents/{doc_id}/tags/?format=json",
                headers=self._auth_headers(),
            )
            tags = resp.json().get("results", []) if resp.status_code == 200 else []

            # Get cabinets
            resp = await client.get(
                f"{self.base_url}/documents/{doc_id}/cabinets/?format=json",
                headers=self._auth_headers(),
            )
            cabinets = resp.json().get("results", []) if resp.status_code == 200 else []

            return {
                "id": doc_id,
                "label": doc.get("label", ""),
                "document_type": doc.get("document_type", {}).get("label", ""),
                "tags": [t.get("label", "") for t in tags],
                "cabinets": [c.get("label", "") for c in cabinets],
                "datetime_created": doc.get("datetime_created", ""),
            }

    async def sync_document(self, doc_id: int) -> MayanDocument:
        """Full sync: download file + get metadata. Returns a MayanDocument ready for RAG pipeline."""
        file_path = await self.get_document_file(doc_id)
        meta = await self.get_document_metadata(doc_id)
        doc_details = await self.get_document(doc_id)

        return MayanDocument(
            id=doc_id,
            label=meta["label"],
            doc_type=meta["document_type"],
            page_count=doc_details.get("version_active", {}).get("pages_count", 0),
            version_id=doc_details.get("version_active", {}).get("id"),
            file_path=file_path,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Write-back: classify document in Mayan after RAG analysis
    # ------------------------------------------------------------------

    # Map RAG doc_type → Mayan DocumentType label
    _DOC_TYPE_MAP = {
        "research_paper": "Research Paper",
        "research_paper_with_images": "Research Paper",
        "contract": "Contract",
        "contract_with_images": "Contract",
        "invoice": "Invoice",
        "resume": "Resume",
        "technical_manual": "Technical Manual",
        "financial_report": "Financial Report",
        "medical_document": "Medical Document",
        "presentation": "Presentation",
    }

    async def classify_document(
        self,
        doc_id: int,
        rag_doc_type: str,
        tags: list[str] | None = None,
    ) -> dict:
        """Write RAG classification back to Mayan: change document type + add tags.

        Only changes the type if the current type is 'Default' (user hasn't classified it).
        Returns a summary of what was changed.
        """
        token = await self._get_token()
        result = {"doc_id": doc_id, "actions": []}

        async with httpx.AsyncClient(timeout=15) as client:
            # 1. Get current document to check existing type
            doc = await self.get_document(doc_id)
            current_type = doc.get("document_type", {}).get("label", "Default")

            # Only change type if user left it as "Default"
            if current_type == "Default" and rag_doc_type in self._DOC_TYPE_MAP:
                target_label = self._DOC_TYPE_MAP[rag_doc_type]

                # Find the Mayan type ID for this label
                type_id = await self._find_document_type_id(client, target_label)
                if type_id:
                    resp = await client.post(
                        f"{self.base_url}/documents/{doc_id}/type/change/?format=json",
                        headers=self._auth_headers(),
                        json={"document_type_id": type_id},
                    )
                    if resp.status_code in (200, 202, 204):
                        result["actions"].append(f"type: Default → {target_label}")
                        print(f"Mayan doc {doc_id}: type changed to '{target_label}'")
                    else:
                        result["actions"].append(f"type change failed: {resp.status_code}")
            else:
                result["actions"].append(f"type kept: {current_type} (user-assigned)")

            # 2. Add tags (create if needed, then attach via /tags/attach/ endpoint)
            if tags:
                for tag_label in tags[:5]:  # max 5 auto-tags
                    tag_id = await self._find_or_create_tag(client, tag_label)
                    if tag_id:
                        resp = await client.post(
                            f"{self.base_url}/documents/{doc_id}/tags/attach/?format=json",
                            headers=self._auth_headers(),
                            json={"tag": tag_id},
                        )
                        if resp.status_code in (200, 201, 204):
                            result["actions"].append(f"tag: +{tag_label}")
                            print(f"Mayan doc {doc_id}: tag '{tag_label}' attached")

        return result

    async def _find_document_type_id(self, client, label: str) -> int | None:
        """Find a Mayan DocumentType ID by label."""
        resp = await client.get(
            f"{self.base_url}/document_types/?format=json",
            headers=self._auth_headers(),
        )
        if resp.status_code != 200:
            return None
        for dt in resp.json().get("results", []):
            if dt["label"] == label:
                return dt["id"]
        return None

    # ------------------------------------------------------------------
    # Auto-extract metadata via Groq and write to Mayan
    # ------------------------------------------------------------------

    # Fields to extract per doc type
    _METADATA_FIELDS = {
        "research_paper": {
            "authors": "Author names (comma-separated)",
            "publication_year": "Publication year (YYYY)",
            "journal_venue": "Journal or conference name",
            "keywords": "Key topics (comma-separated, max 5)",
        },
        "contract": {
            "parties": "Parties involved (comma-separated)",
            "effective_date": "Effective date (YYYY-MM-DD if found)",
            "jurisdiction": "Governing jurisdiction/state/country",
            "contract_term": "Duration or term of the contract",
        },
        "invoice": {
            "vendor": "Vendor or company name",
            "invoice_number": "Invoice number",
            "amount": "Total amount with currency",
            "due_date": "Due date (YYYY-MM-DD if found)",
        },
        "resume": {
            "candidate_name": "Full name of the candidate",
            "email": "Email address",
            "skills": "Key skills (comma-separated, max 5)",
            "experience_years": "Total years of experience (number)",
        },
    }

    async def extract_and_write_metadata(
        self,
        doc_id: int,
        rag_doc_type: str,
        text_preview: str,
    ) -> dict:
        """Use Groq to extract metadata from document text and write to Mayan."""
        base_type = rag_doc_type.replace("_with_images", "")
        fields = self._METADATA_FIELDS.get(base_type, {})

        # Always extract AI summary
        all_fields = {"ai_summary": "A 2-3 sentence factual summary of the document"}
        all_fields.update(fields)

        if not text_preview.strip():
            return {"doc_id": doc_id, "metadata": {}, "error": "no text to extract from"}

        # Ask Groq to extract all fields at once
        extracted = await self._groq_extract_metadata(text_preview, all_fields)
        if not extracted:
            return {"doc_id": doc_id, "metadata": {}, "error": "groq extraction failed"}

        # Write each field to Mayan
        result = {"doc_id": doc_id, "metadata": extracted, "written": []}
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=15) as client:
            for field_name, value in extracted.items():
                if not value or value.lower() in ("n/a", "not found", "unknown", "none"):
                    continue
                success = await self._write_metadata_field(client, doc_id, field_name, value)
                if success:
                    result["written"].append(f"{field_name}={value}")
                    print(f"Mayan doc {doc_id}: metadata {field_name}='{value}'")

        return result

    async def _groq_extract_metadata(self, text: str, fields: dict) -> dict | None:
        """Call Groq to extract structured metadata from document text."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None

        field_list = "\n".join(f"- {name}: {desc}" for name, desc in fields.items())

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Extract metadata from the document text. "
                                    "Reply ONLY with a JSON object containing the requested fields. "
                                    "If a field cannot be determined, use \"N/A\". "
                                    "Do not include any text outside the JSON."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Extract these fields:\n{field_list}\n\n"
                                    f"DOCUMENT TEXT:\n{text[:3000]}"
                                ),
                            },
                        ],
                        "max_tokens": 500,
                        "temperature": 0,
                    },
                )
                if resp.status_code != 200:
                    return None

                raw = resp.json()["choices"][0]["message"]["content"].strip()
                # Parse JSON from response (handle markdown code blocks)
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                import json
                return json.loads(raw)
        except Exception as e:
            print(f"Groq metadata extraction error: {e}")
            return None

    async def _write_metadata_field(self, client, doc_id: int, field_name: str, value: str) -> bool:
        """Write a single metadata field to a Mayan document."""
        # First find the metadata type ID
        resp = await client.get(
            f"{self.base_url}/metadata_types/?format=json&page_size=50",
            headers=self._auth_headers(),
        )
        if resp.status_code != 200:
            return False

        meta_type_id = None
        for mt in resp.json().get("results", []):
            if mt["name"] == field_name:
                meta_type_id = mt["id"]
                break
        if not meta_type_id:
            return False

        # Check if metadata already exists for this document
        resp = await client.get(
            f"{self.base_url}/documents/{doc_id}/metadata/?format=json",
            headers=self._auth_headers(),
        )
        existing_meta_id = None
        if resp.status_code == 200:
            for m in resp.json().get("results", []):
                if m.get("metadata_type", {}).get("id") == meta_type_id:
                    existing_meta_id = m["id"]
                    break

        if existing_meta_id:
            # Update existing
            resp = await client.put(
                f"{self.base_url}/documents/{doc_id}/metadata/{existing_meta_id}/?format=json",
                headers=self._auth_headers(),
                json={"value": str(value)[:200]},
            )
        else:
            # Create new
            resp = await client.post(
                f"{self.base_url}/documents/{doc_id}/metadata/?format=json",
                headers=self._auth_headers(),
                json={"metadata_type_id": meta_type_id, "value": str(value)[:200]},
            )

        return resp.status_code in (200, 201)

    # ------------------------------------------------------------------
    # Status tags: visual indicators on document cards
    # ------------------------------------------------------------------

    _TAG_PROCESSING = "⏳ Processing"
    _TAG_PROCESSING_COLOR = "#f59e0b"  # yellow
    _TAG_READY = "✅ Indexed"
    _TAG_READY_COLOR = "#10b981"  # green
    _TAG_ERROR = "❌ Index Failed"
    _TAG_ERROR_COLOR = "#ef4444"  # red

    async def set_status_processing(self, doc_id: int) -> None:
        """Add '⏳ Processing' tag, remove '✅ Indexed' if present."""
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=15) as client:
            await self._remove_tag(client, doc_id, self._TAG_READY)
            await self._remove_tag(client, doc_id, self._TAG_ERROR)
            tag_id = await self._find_or_create_tag(client, self._TAG_PROCESSING, self._TAG_PROCESSING_COLOR)
            if tag_id:
                await client.post(
                    f"{self.base_url}/documents/{doc_id}/tags/attach/?format=json",
                    headers=self._auth_headers(),
                    json={"tag": tag_id},
                )

    async def set_status_ready(self, doc_id: int) -> None:
        """Remove '⏳ Processing', add '✅ Indexed'."""
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=15) as client:
            await self._remove_tag(client, doc_id, self._TAG_PROCESSING)
            await self._remove_tag(client, doc_id, self._TAG_ERROR)
            tag_id = await self._find_or_create_tag(client, self._TAG_READY, self._TAG_READY_COLOR)
            if tag_id:
                await client.post(
                    f"{self.base_url}/documents/{doc_id}/tags/attach/?format=json",
                    headers=self._auth_headers(),
                    json={"tag": tag_id},
                )

    async def set_status_error(self, doc_id: int) -> None:
        """Remove '⏳ Processing', add '❌ Index Failed'."""
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=15) as client:
            await self._remove_tag(client, doc_id, self._TAG_PROCESSING)
            tag_id = await self._find_or_create_tag(client, self._TAG_ERROR, self._TAG_ERROR_COLOR)
            if tag_id:
                await client.post(
                    f"{self.base_url}/documents/{doc_id}/tags/attach/?format=json",
                    headers=self._auth_headers(),
                    json={"tag": tag_id},
                )

    async def _remove_tag(self, client, doc_id: int, tag_label: str) -> None:
        """Remove a tag from a document by label."""
        resp = await client.get(
            f"{self.base_url}/documents/{doc_id}/tags/?format=json",
            headers=self._auth_headers(),
        )
        if resp.status_code != 200:
            return
        for tag in resp.json().get("results", []):
            if tag["label"] == tag_label:
                await client.post(
                    f"{self.base_url}/documents/{doc_id}/tags/remove/?format=json",
                    headers=self._auth_headers(),
                    json={"tag": tag["id"]},
                )
                return

    async def _find_or_create_tag(self, client, label: str, color: str = "#3b82f6") -> int | None:
        """Find an existing tag by label, or create it."""
        resp = await client.get(
            f"{self.base_url}/tags/?format=json",
            headers=self._auth_headers(),
        )
        if resp.status_code == 200:
            for tag in resp.json().get("results", []):
                if tag["label"].lower() == label.lower():
                    return tag["id"]

        # Create new tag
        resp = await client.post(
            f"{self.base_url}/tags/?format=json",
            headers=self._auth_headers(),
            json={"label": label, "color": color},
        )
        if resp.status_code in (200, 201):
            return resp.json()["id"]
        return None
