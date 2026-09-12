"""secqa.api: the FastAPI service (SPEC section 8) over the shared pipeline objects.

Public surface: :func:`create_app`, :class:`AppState`, the request / response models and
:func:`problem` (the ``application/problem+json`` builder). Everything else is wiring.
"""

from secqa.api.app import create_app, lifespan
from secqa.api.deps import AppState, get_state
from secqa.api.errors import ApiError, problem
from secqa.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    PageResponse,
    ProblemDetail,
    ReadyResponse,
    SearchRequest,
    SearchResponse,
    SqlRequest,
    VersionResponse,
)

__all__ = [
    "ApiError",
    "AppState",
    "AskRequest",
    "AskResponse",
    "HealthResponse",
    "PageResponse",
    "ProblemDetail",
    "ReadyResponse",
    "SearchRequest",
    "SearchResponse",
    "SqlRequest",
    "VersionResponse",
    "create_app",
    "get_state",
    "lifespan",
    "problem",
]
