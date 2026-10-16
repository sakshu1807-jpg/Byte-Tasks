# Tabular Data Analysis Agent Architecture

An intelligent, multi-agent framework designed to perform automated natural language data analysis on tabular datasets (CSV and Excel). Built using **FastAPI**, **LangChain**, **LangGraph**, **Pandas**, and **Streamlit**, this system routes natural language queries through specialized agentic nodes to produce data-driven insights, statistical summaries, anomaly reports, and charts.

## Key Features & Architecture

### 1. In-Memory Processing Pipeline

* **Stateless & Diskless Ingestion:** Uses FastAPI's `UploadFile` and Python’s `io.BytesIO` to read datasets directly into memory as Pandas DataFrames without saving files to local disk.

* **Dynamic Context Binding:** Tool definitions are dynamically bound to the session's DataFrame context, allowing multiple analytical executions without re-uploading or corrupting global state.

### 2. Multi-Agent LangGraph Workflow

The system uses a 5-role workflow orchestrated with LangGraph:

1. **Data Profiler Node:** Programmatically inspects the DataFrame upon ingestion, extracting structural metadata, column data types, missing value distributions, and sample rows.

2. **Planning & Strategist Node:** Evaluates the user's question against the generated dataset metadata to construct a step-by-step execution plan.

3. **Execution & Analysis Node:** Houses a agent equipped with **5 distinct analysis tools** to compute exact analytical results.

4. **Validation Node (Guardrail):** Evaluates whether the user's query matches the dataset's domain/schema and enforces circuit-breaker retries to prevent infinite agent loops on out-of-scope queries.

5. **Synthesis & Reporting Node:** Takes raw execution logs and formats them into structured Markdown reports featuring statistical key takeaways and base64-encoded visual charts.

## The 5 Distinct Analysis Tools

Rather than relying on generic LLM text generation or simple wrappers, the Execution Agent uses five specialized computational tools:

1. **Statistical Summary & Correlation Matrix Tool:** Computes deep descriptive statistics (mean, median, standard deviation, percentiles, min/max) for numeric features using Pandas `.describe()`, and correlation matrices across numerical fields to uncover hidden linear relationships..

2. **Anomaly & Outlier Detection Tool:** Applies the Interquartile Range (IQR) method to detect and extract statistical anomalies in targeted numerical distributions.

3. **Segmentation & Grouping Tool:** Performs dynamic `groupby` operations with customizable aggregation metrics (`mean`, `sum`, `count`) across categorical and numerical columns.

4. **Data Visualization Generator Tool:** Dynamically constructs plots (scatter, bar, histogram, box plots) using Matplotlib and Seaborn, returning them as base64-encoded PNG strings for rendering.

## Technical Stack

* **Backend Framework:** FastAPI, Uvicorn

* **Orchestration & Agents:** LangChain, LangGraph

* **Data Processing & Analytics:** Pandas, NumPy, SciPy, Scikit-Learn

* **Data Visualization:** Matplotlib, Seaborn

* **Frontend UI:** Streamlit

## Project Structure

```
├── main.py                # FastAPI application endpoints & initialization
├── agent_workflow.py     # LangGraph workflow, nodes, tools, and state schema
├── requirements.txt       # Core project dependencies
└── README.md              # Documentation

```

## Getting Started

### 1. Prerequisites

Ensure you have Python 3.10+ installed on your system.

### 2. Installation

Clone the repository and install the dependencies:

```
# Install required Python packages
pip install -r requirements.txt

```

### 3. Environment Setup

Create a `.env` file in the root directory and add your LLM API credentials (e.g., OpenAI, Google Gemini, or Anthropic):

```
GROQ_API_KEY=your_openai_api_key_here
# or
GOOGLE_API_KEY=your_gemini_api_key_here

```

### 4. Running the FastAPI Backend

Start the FastAPI server using Uvicorn:

```
uvicorn main:app --reload

```

The server will spin up at `http://127.0.0.1:8000`. You can access the interactive Swagger API documentation directly at:
👉 **`http://127.0.0.1:8000/docs`**

## API Usage

### `POST /analyze`

Accepts a file upload along with a natural language prompt.

#### Request Parameters (Multipart Form Data):

* `file`: The `.csv`, `.xlsx`, or `.xls` file.

* `question`: The natural language query (e.g., *"Are there any anomalies in the revenue column and what is the correlation between marketing spend and sales?"*).

* **Circuit Breaker:** State-level retry counters prevent infinite execution loops when presented with ambiguous or out-of-scope user queries.
