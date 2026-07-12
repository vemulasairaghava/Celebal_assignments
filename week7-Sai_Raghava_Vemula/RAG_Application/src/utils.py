"""
utils.py — Shared utility functions for the RAG application.

Provides file-type detection, text sanitization, timing helpers,
and directory management used across multiple modules.
"""

import os
import re
import time
import functools
from typing import Callable, Any

from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def detect_file_type(filename: str) -> str:
    """
    Return the lowercase file extension (e.g. '.pdf').

    Raises
    ------
    ValueError
        If the extension is not in ``SUPPORTED_EXTENSIONS``.
    """
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return ext


def sanitize_text(text: str) -> str:
    """
    Clean extracted text.

    * Replace multiple whitespace with a single space.
    * Strip leading / trailing whitespace.
    * Normalise Unicode dashes and quotes.
    """
    if not text:
        return ""
    # Normalise common Unicode characters
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def timer(func: Callable) -> Callable:
    """Decorator that logs the elapsed wall-clock time of a function call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2f s", func.__name__, elapsed)
        return result

    return wrapper


def ensure_directories() -> None:
    """Create all required application directories if they do not exist."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = [
        os.path.join(base, "data", "uploads"),
        os.path.join(base, "data", "processed"),
        os.path.join(base, "vectorstore"),
        os.path.join(base, "logs"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Application directories verified.")


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
