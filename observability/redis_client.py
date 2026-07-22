"""
Lazy Redis client shared by tracer.py and evaluation/runner.py.

Falls back to `None` when REDIS_URL is not configured or the server is
unreachable — callers must handle a None return by using in-memory state.
"""

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_client: Any = None
_client_initialized = False


def _build_client() -> Optional[Any]:
    """Attempt to build a Redis client from the REDIS_URL environment variable."""
    url = os.getenv("REDIS_URL")
    if not url:
        logger.info("REDIS_URL not set — persistence will fall back to in-memory state.")
        return None

    try:
        import redis  # type: ignore

        client = redis.from_url(url, decode_responses=True)
        client.ping()
        logger.info("Redis client initialised.")
        return client
    except Exception as exc:
        logger.warning("Failed to initialise Redis client: %s", exc)
        return None


def get_client() -> Optional[Any]:
    """Return a cached Redis client, creating it on first call."""
    global _client, _client_initialized
    if not _client_initialized:
        _client = _build_client()
        _client_initialized = True
    return _client
