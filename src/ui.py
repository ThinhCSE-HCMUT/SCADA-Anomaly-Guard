"""Shared UI helpers for dashboard styling."""

import streamlit as st


def inject_global_styles() -> None:
    """Inject the shared visual system used across the app."""
    st.markdown(
        """
        <style>
        :root {
            --ui-bg: #0a0f1e;
            --ui-surface: #111827;
            --ui-surface-soft: rgba(17, 24, 39, 0.88);
            --ui-border: rgba(148, 163, 184, 0.16);
            --ui-border-strong: rgba(45, 212, 191, 0.34);
            --ui-text: #e5edf9;
            --ui-muted: #9aa8bd;
            --ui-accent: #2dd4bf;
            --ui-accent-hover: #5eead4;
            --ui-good: #4ade80;
            --ui-warn: #facc15;
            --ui-danger: #fb7185;
            --ui-shadow: 0 16px 42px rgba(0, 0, 0, 0.28);
            --ui-shadow-hover: 0 20px 52px rgba(0, 0, 0, 0.36);
            --ui-radius: 14px;
        }

        .stApp {
            background: var(--ui-bg);
            color: var(--ui-text);
        }

        .main .block-container {
            padding-top: 1.15rem;
            padding-bottom: 2rem;
            max-width: 1440px;
        }

        @media (max-width: 768px) {
            .main .block-container {
                padding-top: 0.9rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }

        section[data-testid="stSidebar"] {
            background: #070b15;
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }

        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] * {
            color: #e2e8f0;
        }

        section[data-testid="stSidebar"] [data-testid="stButton"] button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(15, 23, 42, 0.5);
            color: #f8fafc;
            box-shadow: none;
            transition: transform 160ms ease, border-color 160ms ease, background-color 160ms ease, box-shadow 160ms ease;
        }

        section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
            transform: translateX(3px);
            border-color: rgba(45, 212, 191, 0.45);
            background: rgba(15, 118, 110, 0.22);
        }

        section[data-testid="stSidebar"] a {
            display: block;
            text-decoration: none;
            border-radius: 12px;
            transition: transform 160ms ease, background-color 160ms ease, border-color 160ms ease;
        }

        section[data-testid="stSidebar"] a:hover {
            transform: translateX(3px);
        }

        div[data-testid="stMetric"] {
            background: var(--ui-surface);
            border: 1px solid var(--ui-border);
            border-radius: var(--ui-radius);
            padding: 0.95rem 1rem;
            box-shadow: var(--ui-shadow);
            color: var(--ui-text) !important;
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }

        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label * {
            color: var(--ui-muted) !important;
            opacity: 1 !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] *,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] div {
            color: var(--ui-text) !important;
            opacity: 1 !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricDelta"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
            color: var(--ui-good) !important;
            opacity: 1 !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricDelta"] svg {
            fill: var(--ui-good) !important;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: var(--ui-shadow-hover);
            border-color: var(--ui-border-strong);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--ui-surface);
            border: 1px solid var(--ui-border);
            border-radius: var(--ui-radius);
            box-shadow: var(--ui-shadow);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
            overflow: hidden;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-2px);
            box-shadow: var(--ui-shadow-hover);
            border-color: var(--ui-border-strong);
        }

        div[data-testid="stButton"] > button,
        a[data-testid="stPageLink"],
        div[data-testid="stPageLink"] {
            border-radius: 12px;
            transition: transform 160ms ease, box-shadow 160ms ease, background-color 160ms ease, border-color 160ms ease;
        }

        div[data-testid="stButton"] > button:hover,
        a[data-testid="stPageLink"]:hover,
        div[data-testid="stPageLink"]:hover {
            transform: translateY(-1px);
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--ui-border);
            border-radius: 12px;
            background: var(--ui-surface-soft);
            box-shadow: var(--ui-shadow);
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid var(--ui-border);
            box-shadow: var(--ui-shadow);
        }

        div[data-testid="stMarkdownContainer"],
        div[data-testid="stCaptionContainer"],
        div[data-testid="stHeader"],
        div[data-testid="stHeading"] {
            color: var(--ui-text);
        }

        p,
        span,
        label,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6 {
            color: inherit;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--ui-border);
        }

        button:focus-visible,
        a:focus-visible {
            outline: 2px solid rgba(15, 118, 110, 0.45);
            outline-offset: 2px;
        }

        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
