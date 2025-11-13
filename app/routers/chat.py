"""Chat-style natural language endpoint."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request

from ..config import Settings
from ..core.dataio.schemas import ChatRequest, ChatResponse, RAGQuery
from ..core.nlp.chat_parser import build_chat_query
from ..deps import SettingsDep
from .query import run_rag_pipeline

router = APIRouter(prefix="", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, request: Request, settings: Settings = SettingsDep) -> Dict[str, Any]:
    """Accept a natural-language prompt and return the interpreted query plus answer."""

    locator = getattr(request.app.state, "locator", None)
    try:
        rag_query: RAGQuery = build_chat_query(
            payload,
            default_window_days=settings.default_time_window_days,
            locator=locator,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    answer = run_rag_pipeline(rag_query, request)
    query_payload = rag_query.model_dump() if hasattr(rag_query, 'model_dump') else rag_query
    answer_payload = answer
    return {"query": query_payload, "answer": answer_payload}
