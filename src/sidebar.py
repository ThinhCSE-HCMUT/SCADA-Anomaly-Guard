"""Common sidebar component and shared navigation."""

import streamlit as st

from src.config import APP_DESCRIPTION, APP_TITLE, DEFAULT_MODEL
from src.ui import inject_global_styles


def render_sidebar():
    """Render the shared sidebar shell and return the active model name."""
    inject_global_styles()
    st.session_state.setdefault("selected_model", DEFAULT_MODEL)
    selected_model = st.session_state["selected_model"]
    brand_label = APP_TITLE.split(" - ")[0]

    with st.sidebar:
        st.image("assets/logo.png", use_container_width=True)

        if st.button(brand_label, type="primary", width="stretch", icon=":material/home:"):
            st.switch_page("app.py")

        st.caption(APP_DESCRIPTION)
        st.caption(f"Active model: {selected_model}")
        st.divider()

        st.subheader("Navigation")
        st.page_link("pages/01_Overview.py", label="Overview", width="stretch")
        st.page_link("pages/02_Real-time_Monitor.py", label="Real-time Monitor", width="stretch")
        st.page_link("pages/05_Model_Testing_and_Comparison.py", label="Model Testing & Comparison", width="stretch")
        st.page_link("pages/06_Alerts_Logs.py", label="Alerts & Logs", width="stretch")

    return selected_model
