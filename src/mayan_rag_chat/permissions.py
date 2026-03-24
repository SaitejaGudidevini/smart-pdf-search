from django.utils.translation import gettext_lazy as _

from mayan.apps.permissions.classes import PermissionNamespace

namespace = PermissionNamespace(
    label=_('RAG Chat'), name='rag_chat'
)

permission_rag_chat_use = namespace.add_permission(
    label=_('Use RAG Chat'), name='rag_chat_use'
)

permission_rag_index = namespace.add_permission(
    label=_('Index documents for RAG'), name='rag_index'
)
