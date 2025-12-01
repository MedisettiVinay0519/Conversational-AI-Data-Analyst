# Autonomous Data Analyst Agent (LangGraph) 📊

An intelligent agent built with **LangGraph** and **LangChain** that autonomously executes a complete data analysis pipeline and generates insightful reports.

## ✨ Project Description

This agent orchestrates a full, end-to-end data analysis pipeline, ensuring systematic, step-by-step execution:

Autonomous Data Analyst agent built with **LangGraph** and **LangChain**. It orchestrates a full data pipeline: data loading, missing value handling (imputation), comprehensive EDA plotting (histograms, heatmaps), simple ML modeling (RandomForest), and generation of an insightful, LLM-powered summary report and PDF.

***

## 🧠 LangGraph Workflow Architecture (The "Single Flow" DAG)

The entire analysis is managed by a **LangGraph StateGraph**, which defines a reliable, sequential **Directed Acyclic Graph (DAG)**. This "single flow" architecture moves the data deterministically through five key stages, enforcing order and ensuring no step is missed.

Here's a visual representation of the workflow:


![Autonomous Data Analyst Workflow](image/Screenshot%202025-12-01%20130004.png)

### Flow Nodes:

1.  **`load_data`**: Loads the initial dataset.
2.  **`handle_missing_values`**: Cleans the data (e.g., median/mode imputation).
3.  **`perform_eda`**: Generates visual plots and extracts exploratory insights.
4.  **`run_ml_model`**: Builds a simple benchmark model (RandomForest).
5.  **`generate_llm_summary`**: Synthesizes all reports into a final, human-readable summary.

***

## 🛠️ Setup and Installation

### 1. Prerequisites

* Python 3.9+
* An LLM API Key (e.g., OpenAI or similar)

### 2. Dependencies

Install the required libraries:

```bash
pip install langchain langgraph langchain-core langchain-openai pandas numpy scikit-learn matplotlib seaborn
