import os
import pandas as pd
from typing import TypedDict, Any, Optional, Dict, List
from pandas import DataFrame
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tools import tools_for_dataframe
from dotenv import load_dotenv
load_dotenv()

MAX_VALIDATION_FAILURES = 3

class AgentState(TypedDict):
    df: pd.DataFrame
    file_name: str
    profile_summary: Dict[str, Any]
    user_question: str
    execution_plan: List[str]
    tool_outputs: Dict[str, Any]
    retry_count: int
    validation_status: bool
    final_response: str

async def data_overview_node(state: AgentState) -> AgentState:

    df = state["df"]
    data_type_dict = df.dtypes.astype(str).to_dict()
    missing_value_dict = df.isnull().sum().to_dict()
    profile = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": data_type_dict,
        "missing_values": missing_value_dict,
        "sample_data": df.head(3).to_dict(orient="records")
    }
    
    return {"profile_summary": profile}

async def strategist_node(state: AgentState) -> AgentState:

    profile = state["profile_summary"]
    question = state["user_question"]
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a master data science strategist. Given the dataset profile and "
            "user question, create a concise step-by-step execution plan. "
            "Return only a valid JSON array of strings, for example "
            '["Inspect the relevant columns", "Summarize the matching data"]. '
            "Do not include Markdown, code fences, or prose outside the JSON array.",
        ),
        ("human", "Dataset Profile:\n{profile}\n\nUser Question:\n{question}")
    ])

    llm = ChatGroq(
        model= 'openai/gpt-oss-120b',
        temperature= 0.4,
        api_key= os.getenv('GROQ_API_KEY')
    )

    parser = JsonOutputParser()
    chain: Runnable = prompt | llm | parser

    execution_plan = await chain.ainvoke({"profile": profile, "question": question})
    if not isinstance(execution_plan, list) or not all(
        isinstance(step, str) for step in execution_plan
    ):
        raise ValueError("The strategist must return a JSON array of plan-step strings.")
    
    return {"execution_plan": execution_plan}

async def execution_node(state: AgentState) -> AgentState:

    question = state['user_question']
    df = state['df']

    llm = ChatGoogleGenerativeAI(
        model = 'gemini-3.5-flash-lite',
        api_key = os.getenv('GOOGLE_API_KEY')
    )

    tools_list = tools_for_dataframe(df)

    agent_prompt = """
You are an expert Data Science and Analytics Execution Agent. Your task is to accurately answer natural language questions about a tabular dataset by methodically utilizing your specialized tools.

Available Tools and When to Use Them:
1. **statistical_summary_and_correlation_matrix_tool**: Use this when the user asks for distributions, descriptive stats, averages, spread, or general summaries of numeric columns,
    or relationships, dependencies, or correlations between multiple numeric columns.
2. **anomaly_detection_tool**: Use this when the user asks to find outliers, anomalies, or extreme values in a specific numeric column.
3. **segmentation_grouping_tool**: Use this when the user wants to compare groups (e.g., average sales by product category, sum of revenue by region).
4. **data_visualization_tool**: Use this when the user requests a chart, plot, or visual representation (scatter, bar, histogram, boxplot) of the data.

Execution Rules:
- **Match Exact Names**: Always use exact column names from the dataset schema. 
- **Tool-First Approach**: Do not calculate or guess numbers manually. Always delegate computational or analytical tasks to your tools.
- **Self-Correction**: If a tool returns an error or missing column warning, read the error message carefully, fix your argument parameters, and try the tool call again.
- **Factual Synthesis**: Base your responses strictly on the numerical results or metadata returned by your tools.

CRITICAL RULE: If the user's question cannot be answered using the columns, data types, or contents of this dataset, do NOT call any tools. 
Immediately respond with: 'I am sorry, but this question cannot be answered using the provided dataset.'
"""
    analysis_agent = create_agent(
        model= llm,
        tools = tools_list,
        system_prompt= agent_prompt
    )

    try:
        agent_input = {
            "messages": [
                ("human", f"Dataset shape: {df.shape}\nColumns: {list(df.columns)}\n\nExecution Plan: {state['execution_plan']}\n\nUser Question: {question}")
            ]
        }
        response = analysis_agent.invoke(agent_input)
        final_output = response['messages'][-1].content

    except:
        final_output = "I am sorry, but that question appears to be outside the scope of the provided dataset."

    return {
        "tool_outputs" : {'execution_result' : final_output}
    }

async def validation_node(state: AgentState) -> AgentState:
    question = state["user_question"]
    execution_result = state["tool_outputs"].get("execution_result", "")
    retry_count = state.get('retry_count', 0)

    validation_prompt = f"""
    User Question: {question}
    Agent Execution Result: {execution_result}
    
    Does the execution result successfully answer the user's question? 
    Reply with a single 'YES' if it answers the question well, or a single 'NO' if it is incomplete or hit an error.
    """
    llm = ChatGroq(
        model= 'openai/gpt-oss-120b',
        temperature= 0.35,
        api_key= os.getenv('GROQ_API_KEY')
    )
    validation_response = await llm.ainvoke(validation_prompt)
    val_ans = validation_response.content.strip().upper()
    val_status = val_ans == "YES"
    updated_retry_count = retry_count if val_status else retry_count + 1

    return {
        "validation_status": val_status,
        "retry_count": updated_retry_count,
    }

def router(state: AgentState):
    """Decides whether to re-execute the execution plan based on the boolean value of validation-status"""

    if state['validation_status'] or state.get("retry_count", 0) >= MAX_VALIDATION_FAILURES:
        return "final_report"
    return "execution"

async def report_node(state: AgentState) -> AgentState:

    question = state["user_question"]
    final_result = state["tool_outputs"].get("execution_result", "")
    report_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert Data Presentation Agent. Synthesize the raw analytical findings into a clear, 
        structured, and professional response using Markdown tables, bullet points, and key takeaways."""),
        ("human", "User Question: {question}\n\nAnalytical Findings:\n{result}")
    ])

    llm = ChatGoogleGenerativeAI(
            model = 'gemini-3.5-flash-lite',
            api_key = os.getenv('GOOGLE_API_KEY')
        )

    chain: Runnable = report_prompt | llm
    output = await chain.ainvoke(
        {
            'question': question,
            'result': final_result
        }
    )
    return {
        'final_response': output
    }

graph = StateGraph(AgentState)
graph.add_node("data_overview", data_overview_node)
graph.add_node("stratergy", strategist_node)
graph.add_node("execution", execution_node)
graph.add_node("validation", validation_node)
graph.add_node("final_report", report_node)

graph.add_edge(START, "data_overview")
graph.add_edge("data_overview", "stratergy")
graph.add_edge("stratergy", "execution")
graph.add_edge("execution", "validation")
graph.add_conditional_edges("validation", router)
graph.add_edge("final_report", END)

app_graph = graph.compile()