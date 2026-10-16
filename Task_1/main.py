import io
import pandas as pd

#Fast API components
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import Field

# Agents Compoents

from state import app_graph


app = FastAPI(title= "Data Analyzer Agent")

@app.get("/health")
def stay_alive():
    return {"messge": "Server is running nicely!"}

@app.post('/upload_file')
async def upload_file_and_analyze(user_question: str, file: UploadFile = File(...)):

    file_name = file.filename
    file_ext = file_name.split('.')[-1]

    if not file_ext in ['csv', 'xlsx', 'xls']:
        raise HTTPException(status_code=400, detail="Only CSV or Excel files are supported.")

    file_content = await file.read()
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content))
        else:
            df = pd.read_excel(io.BytesIO(file_content))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing file: {str(e)}")
        
    user_question = user_question.strip()
    initial_state = {
            "df": df,
            "file_name": file_name,
            "profile_summary": {},
            "user_question": user_question,
            "execution_plan": [],
            "tool_outputs": {},
            "retry_count": 0,
            "validation_status": True,
            "final_response": ""
        }

    try:
        final_state = await app_graph.ainvoke(initial_state)

        if final_state['final_response']:
            response_content = final_state['final_response']
        else:
            response_content = "No response has been generated."

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Agent execution error: {str(e)}"
        )

    return {
        "status": "success",
        "file_name": file_name,
        "question": user_question,
        "analysis_result": response_content
    }   
