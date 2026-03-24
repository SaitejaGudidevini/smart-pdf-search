#!/usr/bin/env bash

set -euo pipefail

MAYAN_CONTAINER="${MAYAN_CONTAINER:-mayan-app}"
RAG_SIDECAR_URL="${RAG_SIDECAR_URL:-http://rag-sidecar:8080}"
WORKFLOW_LABEL="${WORKFLOW_LABEL:-RAG Post OCR Indexing}"
WORKFLOW_INTERNAL_NAME="${WORKFLOW_INTERNAL_NAME:-rag_post_ocr_indexing}"
DOCUMENT_TYPE_LABEL="${DOCUMENT_TYPE_LABEL:-Default}"

docker exec "${MAYAN_CONTAINER}" /bin/sh -lc "DJANGO_SETTINGS_MODULE=mayan.settings /opt/mayan-edms/bin/python <<'PY'
import json
import django

django.setup()

from mayan.apps.document_states.literals import WORKFLOW_ACTION_ON_ENTRY
from mayan.apps.document_states.models import (
    Workflow, WorkflowState, WorkflowStateAction, WorkflowTransition
)
from mayan.apps.documents.models import DocumentType
from mayan.apps.events.models import StoredEventType

workflow_label = '${WORKFLOW_LABEL}'
workflow_internal_name = '${WORKFLOW_INTERNAL_NAME}'
document_type_label = '${DOCUMENT_TYPE_LABEL}'
rag_sidecar_url = '${RAG_SIDECAR_URL}'

workflow, workflow_created = Workflow.objects.get_or_create(
    internal_name=workflow_internal_name,
    defaults={
        'label': workflow_label,
        'auto_launch': True,
        'ignore_completed': True,
    }
)

workflow.label = workflow_label
workflow.auto_launch = True
workflow.ignore_completed = True
workflow.save()

queued_state, queued_state_created = WorkflowState.objects.get_or_create(
    workflow=workflow,
    label='Queued for OCR trigger',
    defaults={'initial': True, 'completion': 0}
)
if not queued_state.initial or queued_state.completion != 0:
    queued_state.initial = True
    queued_state.completion = 0
    queued_state.save()

indexed_state, indexed_state_created = WorkflowState.objects.get_or_create(
    workflow=workflow,
    label='Indexed by RAG',
    defaults={'initial': False, 'completion': 100}
)
if indexed_state.initial or indexed_state.completion != 100:
    indexed_state.initial = False
    indexed_state.completion = 100
    indexed_state.save()

transition, transition_created = WorkflowTransition.objects.get_or_create(
    workflow=workflow,
    label='OCR complete to indexed',
    defaults={
        'origin_state': queued_state,
        'destination_state': indexed_state,
    }
)
if (
    transition.origin_state_id != queued_state.pk
    or transition.destination_state_id != indexed_state.pk
):
    transition.origin_state = queued_state
    transition.destination_state = indexed_state
    transition.save()

event_type = StoredEventType.objects.get(name='ocr.document_version_finish')
trigger, trigger_created = transition.trigger_events.get_or_create(
    event_type=event_type
)

document_type = DocumentType.objects.get(label=document_type_label)
workflow.document_types.add(document_type)

backend_data = {
    'url': f'{rag_sidecar_url}/api/mayan/webhook',
    'method': 'POST',
    'headers': '{\"Content-Type\": \"application/json\"}',
    'payload': '{\"document_id\": {{ workflow_instance.document.pk }}, \"event\": \"ocr.document_version_finish\"}',
    'timeout': '10',
    'verify_certificate': False,
    'response_store': True,
    'response_store_name': 'rag_index_response',
}

action, action_created = WorkflowStateAction.objects.get_or_create(
    state=indexed_state,
    label='Post OCR RAG webhook',
    defaults={
        'backend_path': 'mayan.apps.document_states.workflow_actions.HTTPAction',
        'enabled': True,
        'when': WORKFLOW_ACTION_ON_ENTRY,
    }
)
action.backend_path = 'mayan.apps.document_states.workflow_actions.HTTPAction'
action.enabled = True
action.when = WORKFLOW_ACTION_ON_ENTRY
action.set_backend_data(backend_data)
action.save()

print(json.dumps({
    'workflow': {
        'id': workflow.pk,
        'created': workflow_created,
        'label': workflow.label,
        'auto_launch': workflow.auto_launch,
        'document_types': list(workflow.document_types.values_list('label', flat=True))
    },
    'states': [
        {'id': queued_state.pk, 'created': queued_state_created, 'label': queued_state.label},
        {'id': indexed_state.pk, 'created': indexed_state_created, 'label': indexed_state.label}
    ],
    'transition': {
        'id': transition.pk,
        'created': transition_created,
        'label': transition.label
    },
    'trigger': {
        'id': trigger.pk,
        'created': trigger_created,
        'event_type': trigger.event_type.name
    },
    'action': {
        'id': action.pk,
        'created': action_created,
        'label': action.label,
        'backend_data': action.get_backend_data()
    }
}, indent=2))
PY"
