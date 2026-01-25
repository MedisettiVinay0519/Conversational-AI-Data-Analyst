import os
import pandas as pd
from langchain_core.messages import HumanMessage

from ai_agent import AppState, llm
from tools import (
    build_simple_ml_model,
    basic_eda,
    generate_plots,
    build_summary_prompt,
    build_report_text,
    infer_task_and_target
)

# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data_node(state: AppState) -> AppState:
    if isinstance(state.df, str):
        try:
            state.df = pd.read_csv(state.df)

            # 🔥 CLEAN COLUMN NAMES ONCE
            state.df.columns = (
                state.df.columns
                .str.strip()        # remove leading/trailing spaces
                .str.replace(" ", "_")  # replace spaces with _
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    return state



# --------------------------------------------------
# Handle missing values
# --------------------------------------------------

def handle_missing_node(state: AppState) -> AppState:
    if state.df is None:
        return state

    df = state.df.copy()
    state.missing_info = df.isnull().sum().to_dict()

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode().iloc[0], inplace=True)

    state.cleaned_df = df
    return state


# --------------------------------------------------
# ML insights
# --------------------------------------------------

def ml_insights_node(state: AppState) -> AppState:
    df_for_ml = state.cleaned_df if state.cleaned_df is not None else state.df

    if df_for_ml is None:
        state.ml_info = {"error": "No data available for ML."}
        return state

    # 🔥 AUTO-INFER if user input is missing or invalid
    if (
        not state.target_column
        or state.target_column not in df_for_ml.columns
        or not state.task_type
    ):
        inferred_task, inferred_target = infer_task_and_target(df_for_ml)

        if inferred_task is None:
            state.ml_info = {
                "error": "No suitable target found. Only EDA performed."
            }
            return state

        state.task_type = inferred_task
        state.target_column = inferred_target

    state.ml_info = build_simple_ml_model(
        df=df_for_ml,
        task_type=state.task_type,
        target_column=state.target_column,
    )

    if state.ml_info.get("extra_plots"):
        state.plots.extend(state.ml_info["extra_plots"])

    return state


    



# --------------------------------------------------
# Plot generation
# --------------------------------------------------

def plot_node(state: AppState) -> AppState:
    df_for_plots = state.cleaned_df if state.cleaned_df is not None else state.df

    if df_for_plots is None:
        state.plots = []
        return state

    state.plots = generate_plots(df_for_plots)
    return state


# --------------------------------------------------
# LLM summary
# --------------------------------------------------

def llm_summary_node(state: AppState) -> AppState:
    df_for_summary = state.cleaned_df if state.cleaned_df is not None else state.df

    if df_for_summary is None:
        state.summary = "No data available for summary."
        return state

    user_question = state.messages[-1].content

    eda = basic_eda(df_for_summary)
    state.eda_results = eda

    prompt = build_summary_prompt(
        user_question=user_question,
        missing_info=state.missing_info,
        eda=eda,
        ml_info=state.ml_info,
        plots=state.plots,
    )

    response = llm.invoke(prompt)
    state.summary = response.content
    state.messages.append(HumanMessage(content=state.summary))

    return state

# --------------------------------------------------
# Save report as TXT (no PDF dependency)
# --------------------------------------------------

def report_node(state: AppState) -> AppState:
    if state.summary is None:
        return state

    report_text = build_report_text(state)

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "data_analysis_report.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    state.report_path = out_path
    return state
