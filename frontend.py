import streamlit as st
import pandas as pd
import requests

BACKEND_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(
    page_title="Conversational AI Data Analyst",
    layout="wide"
)

st.title("📊 Conversational AI Data Analyst")

# -------------------------
# Upload CSV
# -------------------------
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # User Inputs
    # -------------------------
    task_type = st.selectbox(
        "Select Machine Learning Task",
        ["classification", "regression"]
    )

    target_column = st.selectbox(
        "Select Target Column",
        df.columns.tolist()
    )

    question = st.text_input(
        "Ask a question about your dataset",
        value="Analyze this dataset and give insights"
    )

    # -------------------------
    # Run Analysis
    # -------------------------
    if st.button("Run Analysis 🚀"):
        with st.spinner("Running AI analysis..."):
            files = {
                "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
                "text/csv"
                            )
            }

            data = {
                "task_type": task_type,
                "target_column": target_column,
                "question": question,
            }

            response = requests.post(
            BACKEND_URL,
            files=files,
                data=data)

        # -------------------------
        # Backend Safety Checks
        # -------------------------
        if response.status_code != 200:
            st.error("❌ Backend returned an error")
            st.subheader("Raw backend response")
            st.code(response.text)
            st.stop()

        try:
            result = response.json()
        except Exception:
            st.error("❌ Backend did not return JSON")
            st.subheader("Raw backend response")
            st.code(response.text)
            st.stop()

        # -------------------------
        # Display Results
        # -------------------------
        st.success("Analysis completed successfully")

        st.subheader("🧠 AI Summary")
        st.write(result.get("summary", "No summary available"))

        st.subheader("📈 ML Results")
        st.json(result.get("ml_info", {}))

        plots = result.get("plots", [])
        if plots:
            st.subheader("📊 Generated Plots")
            for plot in plots:
                st.image(plot)

        report_path = result.get("report_path")
        if report_path:
            st.subheader("📄 Report Path")
            st.code(report_path)
