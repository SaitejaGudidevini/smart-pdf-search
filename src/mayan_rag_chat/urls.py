from django.urls import re_path

from .views import (
    ChatView, ChunkInspectView, IndexAllView, IndexDocumentView, SearchAPIView,
    NudgeView, SummarizeView,
)

app_name = 'rag_chat'

urlpatterns = [
    re_path(r'^chat/$', ChatView.as_view(), name='chat_view'),
    re_path(r'^chunks/$', ChunkInspectView.as_view(), name='chunk_inspect_home_view'),
    re_path(r'^chunks/(?P<document_id>\d+)/$', ChunkInspectView.as_view(), name='chunk_inspect_view'),
    re_path(r'^index/$', IndexAllView.as_view(), name='index_all_view'),
    re_path(r'^index/(?P<document_id>\d+)/$', IndexDocumentView.as_view(), name='index_document_view'),
    re_path(r'^api/search/$', SearchAPIView.as_view(), name='search_api'),
    re_path(r'^nudge/(?P<document_id>\d+)/$', NudgeView.as_view(), name='nudge_view'),
    re_path(r'^summarize/$', SummarizeView.as_view(), name='summarize_view'),
]

api_urls = []
