import os
import textwrap

import requests
import streamlit as st


def _post_json(url: str, payload: dict, timeout_s: int = 180) -> tuple[int, dict | str]:
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        ct = (r.headers.get("content-type") or "").lower()
        if "application/json" in ct:
            return r.status_code, r.json()
        return r.status_code, r.text
    except Exception as e:
        return 0, f"Request failed: {e}"


st.set_page_config(page_title="MedAssist.AI (Cortex)", layout="wide")
st.title("MedAssist.AI — Cortex-backed Clinician Assistant")

api_base = st.sidebar.text_input(
    "FastAPI base URL",
    value=os.environ.get("MEDASSIST_API_BASE", "http://127.0.0.1:8000"),
    help="Where your FastAPI server is running.",
)

mode = st.sidebar.selectbox(
    "Mode",
    options=[
        ("Cortex primary (/ask)", "ask"),
        ("Cortex-only MedRAG (/ask-cortex)", "ask-cortex"),
        ("Compare Gemini vs Cortex (/ask-both)", "ask-both"),
    ],
    format_func=lambda x: x[0],
)

brief = st.sidebar.checkbox("Brief answer", value=True, help="Applies to /ask and /ask-both.")

default_q = "fever vomiting"
question = st.text_area("Clinical question", value=default_q, height=120)

col_a, col_b = st.columns([1, 3])
with col_a:
    run = st.button("Ask", type="primary")
with col_b:
    st.caption(
        "Tip: For symptom-style prompts, try short phrases like "
        + textwrap.dedent("`fever vomiting headache`")
        + " to get stronger retrieval from the symptom map."
    )

if run:
    endpoint = mode[1]
    url = api_base.rstrip("/") + "/" + endpoint

    payload: dict = {"question": question}
    if endpoint in ("ask", "ask-both"):
        payload["brief"] = bool(brief)

    with st.spinner("Querying..."):
        status, data = _post_json(url, payload)

    st.subheader("Result")
    st.write(f"**HTTP status:** {status}")

    if isinstance(data, str):
        st.code(data)
    else:
        if endpoint == "ask":
            st.write(f"**provider:** `{data.get('provider')}`")
            if data.get("fallback_from"):
                st.warning(f"Fell back from {data.get('fallback_from')} to {data.get('provider')}")
            if data.get("cortex_error"):
                st.expander("Cortex error (if any)", expanded=False).code(data.get("cortex_error"))
            st.markdown(data.get("answer", ""))
        elif endpoint == "ask-cortex":
            st.markdown(data.get("answer", ""))
        else:
            left, right = st.columns(2)
            with left:
                st.markdown("### Gemini")
                st.markdown(data.get("answer_gemini", ""))
            with right:
                st.markdown("### Cortex")
                st.markdown(data.get("answer_cortex", ""))

