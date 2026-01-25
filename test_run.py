from workflow import app
from langchain_core.messages import HumanMessage

csv_path = r"C:\Users\LENOVO\Downloads\university_admission.csv"

result = app.invoke({
    "messages": [HumanMessage(content="Analyze churn drivers")],
    "df": csv_path,
    "task_type": "classification",
    "target_column": "Churn",
})

print(result["ml_info"])
print(result["summary"])
