from django.utils.translation import gettext_lazy as _

from mayan.apps.documents.permissions import permission_document_view
from mayan.apps.navigation.links import Link

from .icons import icon_rag_chat, icon_rag_chunks


link_rag_chat = Link(
    icon=icon_rag_chat,
    permission=permission_document_view,
    text=_(message='RAG Chat'),
    view='rag_chat:chat_view'
)

link_rag_chunk_inspector = Link(
    icon=icon_rag_chunks,
    permission=permission_document_view,
    text=_(message='Chunk Inspector'),
    view='rag_chat:chunk_inspect_home_view'
)

link_rag_summarize = Link(
    icon=icon_rag_chat,
    permission=permission_document_view,
    text=_(message='Executive Briefing'),
    view='rag_chat:summarize_view'
)

link_document_rag_chunks = Link(
    args='resolved_object.pk',
    icon=icon_rag_chunks,
    permission=permission_document_view,
    text=_(message='View RAG Chunks'),
    view='rag_chat:chunk_inspect_view'
)

link_document_nudge = Link(
    args='resolved_object.pk',
    icon=icon_rag_chat,
    permission=permission_document_view,
    text=_(message='AI Insights'),
    view='rag_chat:nudge_view'
)
