"""FastAPI service exposing Layer 1 (intent) and Layer 3 (dispatch projection)."""

from solace.api.app import create_app

__all__ = ["create_app"]
