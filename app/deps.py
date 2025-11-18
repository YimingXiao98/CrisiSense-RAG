"""Common dependency injections for FastAPI routers."""
from __future__ import annotations

from fastapi import Depends

from .config import Settings, get_settings


def get_app_settings() -> Settings:
    """Return cached application settings."""

    return get_settings()


SettingsDep = Depends(get_app_settings)
