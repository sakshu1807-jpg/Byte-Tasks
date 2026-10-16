### THIS FRONTEND (streamlit) CODE IS ENTIRELY MADE BY USING GENERATIVE AI.

import asyncio
import hashlib
import io
from typing import Any

import pandas as pd
import streamlit as st

from state import app_graph
from dotenv import load_dotenv
load_dotenv()

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
        else:
            initial_state: dict[str, Any] = {
                "df": dataset,
                "file_name": st.session_state.dataset_name,
                "profile_summary": {},
                "user_question": question.strip(),
                "execution_plan": [],
                "tool_outputs": {},
                "retry_count": 0,
                "validation_status": True,
                "final_response": "",
            }
            try:
                with st.spinner("Analyzing your dataset..."):
                    final_state = asyncio.run(app_graph.ainvoke(initial_state))
            except Exception as exc:
                st.error(
                    "The analysis could not be completed. Check that your "
                    "GROQ_API_KEY and GOOGLE_API_KEY are configured, then try again. "
                    f"Details: {exc}"
                )
            else:
                response = final_state.get("final_response", "")
                if hasattr(response, "content"):
                    response = response.content
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
                    "execution_plan": final_state.get("execution_plan", []),
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