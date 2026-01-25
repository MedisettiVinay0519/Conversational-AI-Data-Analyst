# Conversational AI Data Analyst 🤖📊

A dataset-agnostic AI system that performs:
- Automatic EDA
- Intelligent ML task & target inference
- Classification & regression modeling
- LLM-powered business insights
- FastAPI backend + Streamlit frontend

## 🚀 Features
- Upload **any CSV dataset**
- Auto-detects:
  - Classification vs Regression
  - Target variable
- Handles:
  - Categorical encoding
  - Class imbalance
  - ROC-AUC / RMSE metrics
- Generates:
  - Plots
  - Feature importance
  - Business-style summaries using LLMs

## 🏗️ Architecture
![Project Architecture](image/Screenshot%202025-12-01%20130004.png)


## Start backend
```bash
uvicorn app:api --reload
```
## Start frontend
```bash
streamlit run frontend.py
```
## clone repo
```bash
git clone https://github.com/MedisettiVinay0519/Conversational-AI-Data-Analyst.git
```

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
