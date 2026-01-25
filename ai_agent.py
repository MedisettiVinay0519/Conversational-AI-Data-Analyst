import os
from dotenv import load_dotenv
from typing import Optional, Any, List, Dict, Annotated

from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langchain_groq import ChatGroq

# --------------------------------------------------
# ENV + LLM
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME
)

SYSTEM_PROMPT = """
You are a Conversational AI Data Analyst.
Analyze user queries step-by-step and provide clear insights.
"""

# --------------------------------------------------
# LANGGRAPH STATE
# --------------------------------------------------

class AppState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # conversation
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    # data
    df: Optional[Any] = None
    cleaned_df: Optional[Any] = None

    # 🔥 NEW: user choices
    task_type: Optional[str] = None        # "classification" | "regression"
    target_column: Optional[str] = None    # column name

    # analysis artifacts
    missing_info: Optional[Dict] = None
    eda_results: Optional[Dict] = None
    plots: List[str] = Field(default_factory=list)
    ml_info: Optional[Dict] = None

    # outputs
    summary: Optional[str] = None
    report_path: Optional[str] = None
