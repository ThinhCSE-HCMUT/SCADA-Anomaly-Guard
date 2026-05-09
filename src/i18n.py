"""Small i18n helper for bilingual UI."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.locales.en import TEXT as EN_TEXT
from src.locales.vi import TEXT as VI_TEXT

DEFAULT_LANGUAGE = "en"
LANGUAGES = {
    "en": "English",
    "vi": "Tiếng Việt",
}
TRANSLATIONS = {
    "en": EN_TEXT,
    "vi": VI_TEXT,
}


def ensure_language_state() -> str:
    """Ensure session language exists and is valid."""
    lang = st.session_state.get("language", DEFAULT_LANGUAGE)
    if lang not in TRANSLATIONS:
        lang = DEFAULT_LANGUAGE
    st.session_state["language"] = lang
    return lang


def get_language() -> str:
    """Return active language code."""
    return ensure_language_state()


def set_language(language: str) -> None:
    """Set active language code."""
    st.session_state["language"] = language if language in TRANSLATIONS else DEFAULT_LANGUAGE


def t(key: str, **kwargs: Any) -> str:
    """Translate key for active language."""
    lang = ensure_language_state()
    text = TRANSLATIONS.get(lang, EN_TEXT).get(key, EN_TEXT.get(key, key))
    if kwargs:
        return text.format(**kwargs)
    return text


def label_for_status(status: str) -> str:
    """Translate status-like values with fallback to original string."""
    key = f"status.{status.lower()}"
    return t(key) if key in TRANSLATIONS[ensure_language_state()] or key in EN_TEXT else status

