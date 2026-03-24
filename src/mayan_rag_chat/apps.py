"""Mayan EDMS RAG Chat App — registers as a native Mayan app with URL routing."""

from django.utils.translation import gettext_lazy as _
from django.apps import apps as django_apps

from mayan.apps.app_manager.apps import MayanAppConfig
from mayan.apps.common.menus import menu_main, menu_object


class RAGChatApp(MayanAppConfig):
    app_namespace = 'rag_chat'
    app_url = 'rag_chat'
    has_rest_api = False
    has_tests = False
    name = 'mayan_rag_chat'
    verbose_name = _('RAG Chat')

    def ready(self):
        super().ready()
        from .links import (
            link_document_rag_chunks, link_document_nudge,
            link_rag_chat, link_rag_chunk_inspector, link_rag_summarize,
        )
        Document = django_apps.get_model(app_label='documents', model_name='Document')

        menu_main.bind_links(
            links=(link_rag_chat, link_rag_chunk_inspector, link_rag_summarize), position=75
        )
        menu_object.bind_links(
            links=(link_document_rag_chunks, link_document_nudge), sources=(Document,)
        )
