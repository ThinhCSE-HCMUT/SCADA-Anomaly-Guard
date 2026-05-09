"""Common sidebar component and shared navigation."""

import streamlit as st

from src.config import APP_TITLE, DEFAULT_MODEL
from src.i18n import LANGUAGES, get_language, t
from src.ui import inject_global_styles


def render_sidebar():
    """Render the shared sidebar shell and return the active model name."""
    inject_global_styles()
    language = get_language()
    st.session_state.setdefault("selected_model", DEFAULT_MODEL)
    selected_model = st.session_state["selected_model"]
    brand_label = APP_TITLE.split(" - ")[0]

    with st.sidebar:
        st.image("assets/logo.png", use_container_width=True)

        if st.button(brand_label, type="primary", width="stretch", icon=":material/home:"):
            st.switch_page("app.py")

        st.selectbox(
            t("sidebar.language"),
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index(language),
            format_func=lambda code: LANGUAGES[code],
            key="language",
        )

        st.caption(t("app.description"))
        st.caption(t("sidebar.active_model", model=selected_model))
        st.divider()

        st.subheader(t("sidebar.navigation"))
        st.page_link("pages/01_Overview.py", label=t("nav.overview"), width="stretch")
        st.page_link("pages/02_Real-time_Monitor.py", label=t("nav.monitor"), width="stretch")
        st.page_link("pages/05_Model_Testing_and_Comparison.py", label=t("nav.testing"), width="stretch")
        st.page_link("pages/06_Alerts_Logs.py", label=t("nav.alerts"), width="stretch")

    return selected_model
