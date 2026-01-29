import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from langchain_core.messages import HumanMessage

# -----------------------------
# FastAPI app (MUST be named `app`)
# -----------------------------
app = FastAPI(title="Conversational AI Data Analyst")

# folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}

# -----------------------------
# Main analysis endpoint
# -----------------------------
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    task_type: str = Form(...),       # classification / regression
    target_column: str = Form(...),   # column name
    question: str = Form("Analyze this dataset"),
):
    """
    Upload CSV + choose ML task and target.
    """

    # Save uploaded file
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔥 Lazy import (CRITICAL FIX)
    from workflow import app as langgraph_app

    # Run LangGraph pipeline
    result = langgraph_app.invoke({
        "messages": [HumanMessage(content=question)],
        "df": file_path,
        "task_type": task_type,
        "target_column": target_column,
    })

    return {
        "summary": result.get("summary"),
        "ml_info": result.get("ml_info"),
        "plots": result.get("plots"),
        "report_path": result.get("report_path"),
    }
