"""Thin entry-point that re-exports the FastAPI app from the package."""
from nail_try_on.api.main import app

__all__ = ["app"]
