"""
llm.py — LLM initialisation for the RAG pipeline.

Supports Google Gemini 2.5 Flash (default) and OpenAI GPT-4.1 (fallback).
The provider is selected automatically based on available API keys.
"""

import os

from dotenv import load_dotenv

from src.logger import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


def get_llm(provider: str | None = None):
    """
    Return a LangChain chat model instance.

    Provider selection logic:
    1. If ``provider`` is explicitly given, use it.
    2. If ``GOOGLE_API_KEY`` is set → use Google Gemini.
    3. If ``OPENAI_API_KEY`` is set → use OpenAI GPT-4.1.
    4. Otherwise raise an error.

    Parameters
    ----------
    provider : str | None
        ``"google"`` or ``"openai"``. Auto-detected if ``None``.

    Returns
    -------
    BaseChatModel
        A LangChain chat model with ``temperature=0``.
    """
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    # --- Auto-detect provider ---
    if provider is None:
        if google_key:
            provider = "google"
        elif openai_key:
            provider = "openai"
        else:
            raise EnvironmentError(
                "No LLM API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY in your .env file."
            )

    # --- Instantiate model ---
    if provider == "google":
        if not google_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set in .env.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info("Initialising LLM: Google Gemini 2.5 Flash")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_key,
            temperature=0,
            max_output_tokens=2048,
            timeout=60,
        )

    if provider == "openai":
        if not openai_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in .env.")
        from langchain_openai import ChatOpenAI

        logger.info("Initialising LLM: OpenAI GPT-4.1")
        return ChatOpenAI(
            model="gpt-4.1",
            api_key=openai_key,
            temperature=0,
            max_tokens=2048,
            timeout=60,
        )

    raise ValueError(f"Unknown LLM provider: '{provider}'. Use 'google' or 'openai'.")


def get_provider_name() -> str:
    """Return a human-readable name for the active LLM provider."""
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if google_key:
        return "Google Gemini 2.5 Flash"
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return "OpenAI GPT-4.1"
    return "No API key configured"
