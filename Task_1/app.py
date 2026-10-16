### THIS FRONTEND (streamlit) CODE IS ENTIRELY MADE BY USING GENERATIVE AI.

import hashlib
import io
import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
load_dotenv()


def analyze_with_backend(
    backend_url: str, question: str, file_name: str, file_bytes: bytes
) -> dict:
    boundary = uuid.uuid4().hex
    safe_file_name = file_name.replace("\\", "_").replace("/", "_").replace('"', "")
    content_type = mimetypes.guess_type(safe_file_name)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{safe_file_name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("ascii")
    query = urllib.parse.urlencode({"user_question": question})
    endpoint = f"{backend_url.rstrip('/')}/upload_file?{query}"
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend returned HTTP {exc.code}: {error_body}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Backend returned an unexpected response format.")
    if "analysis_result" not in payload:
        raise RuntimeError(f"Backend response did not include analysis_result: {payload}")
    return payload


st.set_page_config(
    page_title="Byte | Data Analyst",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2.5rem; padding-bottom: 3rem;}
    .hero {padding: 1.6rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg, #11243a, #245c68); color: white; margin-bottom: 1.5rem;}
    .hero h1 {margin: 0 0 .4rem 0; font-size: 2.25rem;}
    .hero p {margin: 0; color: #d5e7e9; font-size: 1.05rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Byte Data Analyst</h1>
      <p>Upload a dataset, explore its contents, and ask questions in plain English.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Your dataset")
backend_url = st.sidebar.text_input(
    "Backend URL",
    value=os.getenv("BACKEND_URL", "http://127.0.0.1:8000"),
    help="Enter your FastAPI base URL, for example https://your-service.onrender.com.",
).strip()
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="The file is read into memory for this session.",
)

st.session_state.setdefault("dataset", None)
st.session_state.setdefault("dataset_name", "")
st.session_state.setdefault("upload_signature", None)
st.session_state.setdefault("analysis", None)

if uploaded_file is None:
    st.session_state.dataset = None
    st.session_state.dataset_name = ""
    st.session_state.upload_signature = None
    st.session_state.analysis = None
    st.info("Upload a CSV or Excel file from the sidebar to get started.")
    st.stop()

file_name = uploaded_file.name
file_bytes = uploaded_file.getvalue()
signature = hashlib.sha256(file_bytes).hexdigest()

if signature != st.session_state.upload_signature:
    st.session_state.dataset = None
    st.session_state.dataset_name = file_name
    st.session_state.upload_signature = signature
    st.session_state.analysis = None

    try:
        if file_name.lower().endswith(".csv"):
            dataset = pd.read_csv(io.BytesIO(file_bytes))
        else:
            dataset = pd.read_excel(io.BytesIO(file_bytes))
    except Exception as exc:
        st.session_state.upload_signature = None
        st.error(f"Could not read {file_name}: {exc}")
        st.stop()

    if dataset.empty:
        st.warning("This file contains no data rows. Upload a non-empty dataset to continue.")
        st.stop()

    st.session_state.dataset = dataset
    st.session_state.dataset_name = file_name

dataset: pd.DataFrame = st.session_state.dataset
st.caption(f"Loaded **{st.session_state.dataset_name}**")

row_metric, column_metric, missing_metric = st.columns(3)
row_metric.metric("Rows", f"{len(dataset):,}")
column_metric.metric("Columns", f"{len(dataset.columns):,}")
missing_metric.metric("Missing values", f"{int(dataset.isna().sum().sum()):,}")

preview_tab, schema_tab, analysis_tab = st.tabs(
    ["Data preview", "Column details", "Ask Byte"]
)

with preview_tab:
    st.subheader("A look at your data")
    st.dataframe(dataset.head(100), use_container_width=True, hide_index=True)
    if len(dataset) > 100:
        st.caption("Showing the first 100 rows.")

with schema_tab:
    st.subheader("Columns and data types")
    schema = pd.DataFrame(
        {
            "Column": dataset.columns.astype(str),
            "Data type": dataset.dtypes.astype(str).values,
            "Missing values": dataset.isna().sum().values,
            "Unique values": dataset.nunique(dropna=True).values,
        }
    )
    st.dataframe(schema, use_container_width=True, hide_index=True)

with analysis_tab:
    st.subheader("Ask a question about your data")
    st.caption(
        "Try questions about summaries, correlations, outliers, group comparisons, or charts."
    )

    with st.form("analysis_form"):
        question = st.text_area(
            "Your question",
            placeholder="For example: Which category has the highest average sales?",
            height=100,
        )
        submitted = st.form_submit_button(
            "Analyze dataset",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not question.strip():
            st.warning("Enter a question before starting the analysis.")
        elif not backend_url:
            st.warning("Enter the FastAPI backend URL in the sidebar.")
        else:
            try:
                with st.spinner("Analyzing your dataset..."):
                    backend_response = analyze_with_backend(
                        backend_url,
                        question.strip(),
                        uploaded_file.name,
                        file_bytes,
                    )
            except Exception as exc:
                st.error(f"The backend analysis request failed. Details: {exc}")
            else:
                response = backend_response["analysis_result"]
                if isinstance(response, dict) and "content" in response:
                    response = response["content"]
                if isinstance(response, list):
                    text_blocks = [
                        block["text"]
                        for block in response
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                    ]
                    if text_blocks:
                        response = "\n\n".join(text_blocks)
                st.session_state.analysis = {
                    "question": question.strip(),
                    "response": response,
                    "execution_plan": backend_response.get("execution_plan", []),
                }

    analysis = st.session_state.analysis
    if analysis:
        st.divider()
        st.markdown(f"**Question:** {analysis['question']}")
        response = analysis["response"]
        if isinstance(response, str):
            st.markdown(response)
        else:
            st.write(response)

        execution_plan = analysis["execution_plan"]
        if execution_plan:
            with st.expander("View the analysis plan"):
                if isinstance(execution_plan, list):
                    for step in execution_plan:
                        st.write(f"- {step}")
                else:
                    st.write(execution_plan)