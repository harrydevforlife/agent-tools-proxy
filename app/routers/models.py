"""
routers/models.py — GET /v1/models and GET /v1/models/{model}

Returns the configured model from settings. This is a static response
(not proxied from the backend) so it works with all adapter types.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.openai import ModelListResponse, ModelObject

router = APIRouter()


def _model_object() -> ModelObject:
    return ModelObject(id=settings.llm_model)


@router.get("/v1/models")
async def list_models() -> ModelListResponse:
    return ModelListResponse(data=[_model_object()])


@router.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str) -> ModelObject:
    if model_id != settings.llm_model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available: {settings.llm_model}",
        )
    return _model_object()
