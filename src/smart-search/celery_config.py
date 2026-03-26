"""Celery application configuration for Excel background processing.

Uses RabbitMQ as broker (shared with Mayan EDMS) and Redis as result backend.
Both services are already provisioned in docker/docker-compose.yml.

Result backend uses Redis DB 2 to avoid conflicts:
  - DB 0: Mayan EDMS cache
  - DB 1: Mayan Celery results
  - DB 2: RAG Excel worker results
"""

from __future__ import annotations

import os

from celery import Celery

app = Celery("excel_worker")

app.config_from_object(
    {
        "broker_url": os.environ.get(
            "CELERY_BROKER_URL", "amqp://mayan:mayan@rabbitmq:5672/mayan"
        ),
        "result_backend": os.environ.get(
            "CELERY_RESULT_BACKEND", "redis://redis:6379/2"
        ),
        "task_serializer": "json",
        "result_serializer": "json",
        "accept_content": ["json"],
        "task_track_started": True,
        "task_acks_late": True,
        "worker_prefetch_multiplier": 1,
        "task_reject_on_worker_lost": True,
        "task_time_limit": 1800,  # 30 min hard limit per task
        "task_soft_time_limit": 1500,  # 25 min soft limit — allows cleanup
        "result_expires": 86400,  # results expire after 24 hours
    }
)
