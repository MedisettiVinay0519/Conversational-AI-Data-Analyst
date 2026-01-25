import os
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, roc_curve


def generate_plots(df: pd.DataFrame, out_dir: str = "plots") -> List[str]:
    """
    Generate EDA plots for numeric and categorical columns.
    Returns list of saved plot paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])
    ]

    # 1) Histograms
    for col in num_cols:
        out_path = os.path.join(out_dir, f"hist_{col}.png")
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        paths.append(out_path)

    # 2) Boxplots
    for col in num_cols:
        out_path = os.path.join(out_dir, f"box_{col}.png")
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Box Plot of {col}")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        paths.append(out_path)

    # 3) Count plots
    for col in cat_cols:
        if df[col].nunique() <= 20:
            out_path = os.path.join(out_dir, f"count_{col}.png")
            plt.figure()
            sns.countplot(y=df[col])
            plt.title(f"Count Plot of {col}")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            paths.append(out_path)

    # 4) Scatter plots (pairwise)
    if len(num_cols) >= 2:
        for i in range(len(num_cols) - 1):
            x, y = num_cols[i], num_cols[i + 1]
            out_path = os.path.join(out_dir, f"scatter_{x}_vs_{y}.png")
            plt.figure()
            sns.scatterplot(x=df[x], y=df[y])
            plt.title(f"Scatter: {x} vs {y}")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            paths.append(out_path)

    # 5) Correlation heatmap
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        out_path = os.path.join(out_dir, "heatmap_corr.png")
        plt.figure(figsize=(7, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        paths.append(out_path)

    return paths

def build_simple_ml_model(
    df: pd.DataFrame,
    task_type: str,
    target_column: str
) -> Dict:

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ---------------------------
    # Feature separation
    # ---------------------------
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if len(num_features) + len(cat_features) == 0:
        return {"error": "No usable features found."}

    # ---------------------------
    # Preprocessing
    # ---------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ]
    )

    # ---------------------------
    # Train / test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if task_type == "classification" else None
    )

    # ---------------------------
    # Model selection
    # ---------------------------
    if task_type == "classification":
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )
        class_weight_dict = dict(zip(classes, class_weights))

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight=class_weight_dict,
        )

    elif task_type == "regression":
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        )

    else:
        return {"error": "Invalid task_type. Use 'classification' or 'regression'."}

    # ---------------------------
    # Pipeline
    # ---------------------------
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # ---------------------------
    # Metrics
    # ---------------------------
    extra_plots = []
    if task_type == "classification":
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        metrics = {
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
           
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }
        extra_plots = generate_classification_plots(
            y_test,
            y_pred,
            y_proba
        )


    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics = {
            "rmse": float(rmse),
        }

        # ---------------------------
    # Feature importance (FIXED)
    # ---------------------------
    preprocessor_fitted = pipeline.named_steps["preprocessor"]

    num_feature_names = num_features

    cat_feature_names = []
    if cat_features:
        cat_encoder = preprocessor_fitted.named_transformers_["cat"]
        cat_feature_names = cat_encoder.get_feature_names_out(cat_features).tolist()

    feature_names = num_feature_names + cat_feature_names

    importances = pipeline.named_steps["model"].feature_importances_

    feature_importances = sorted(
        [
            {"feature": f, "importance": float(i)}
            for f, i in zip(feature_names, importances)
        ],
        key=lambda d: d["importance"],
        reverse=True,
    )

    return {
        "task_type": task_type,
        "target": target_column,
        "metrics": metrics,
        "n_features": len(feature_names),
        "n_samples": int(len(df)),
        "top_features": feature_importances[:10],
        "extra_plots": extra_plots,
    }


def build_summary_prompt(
    user_question,
    missing_info,
    eda,
    ml_info,
    plots,
):
    task_type = ml_info.get("task_type") if ml_info else None
    target = ml_info.get("target") if ml_info else None
    metrics = ml_info.get("metrics") if ml_info else None
    top_features = ml_info.get("top_features") if ml_info else None

    return f"""
You are a senior data analyst.

The user asked:
{user_question}

--------------------------------
DATA QUALITY
--------------------------------
Missing values (before cleaning):
{missing_info}

--------------------------------
EDA SUMMARY
--------------------------------
{eda}

--------------------------------
ML MODEL RESULTS
--------------------------------
Task type: {task_type}
Target variable: {target}

Model metrics:
{metrics}

Top contributing features:
{top_features}

--------------------------------
GENERATED PLOTS
--------------------------------
{plots}

--------------------------------
INSTRUCTIONS
--------------------------------
Write a clear, business-style summary that explains:

1. What the dataset contains (size, key columns, data types).
2. Data quality issues and how they were handled.
3. Key patterns and insights from EDA.
4. ML results:
   - For classification: explain accuracy vs ROC-AUC and what they imply.
   - For regression: explain RMSE and prediction quality.
5. The business meaning of the most important features.
6. How the plots support the insights.
7. 3–5 concrete next steps for a data or business team.

Use bullet points where appropriate.
Keep the tone professional and actionable.
"""
def basic_eda(df: pd.DataFrame) -> Dict:
    """
    Basic EDA summary used by the LLM.
    """
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
    }


def build_report_text(state) -> str:
    """
    Builds a human-readable analysis report.
    """
    lines = []

    lines.append("AI Data Analyst Report")
    lines.append("=" * 40)
    lines.append("")

    if state.df is not None:
        lines.append(f"Dataset shape: {state.df.shape}")
        lines.append("")

    if state.missing_info is not None:
        lines.append("Missing Values Summary:")
        lines.append(str(state.missing_info))
        lines.append("")

    if state.ml_info is not None:
        lines.append("ML Model Results:")
        lines.append(str(state.ml_info))
        lines.append("")

    lines.append("LLM Summary:")
    lines.append(state.summary)
    lines.append("")

    if state.plots:
        lines.append(f"Generated {len(state.plots)} plots.")

    return "\n".join(lines)


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List


def generate_classification_plots(
    y_true,
    y_pred,
    y_proba,
    out_dir="plots"
) -> List[str]:

    os.makedirs(out_dir, exist_ok=True)
    paths = []

    # -------------------------
    # Encode labels safely
    # -------------------------
    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true)
    y_pred_enc = le.transform(y_pred)

    # positive class = encoded value 1
    pos_label = 1

    # -------------------------
    # Confusion Matrix
    # -------------------------
    cm = confusion_matrix(y_true_enc, y_pred_enc)

    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    paths.append(cm_path)

    # -------------------------
    # ROC Curve
    # -------------------------
    fpr, tpr, _ = roc_curve(
        y_true_enc,
        y_proba,
        pos_label=pos_label
    )

    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    roc_path = os.path.join(out_dir, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    paths.append(roc_path)

    return paths
def infer_task_and_target(df: pd.DataFrame):
    """
    Infer ML task type and target column automatically.
    """
    # Prefer classification targets
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() <= 10:
            return "classification", col

    # Else pick numeric column for regression
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        return "regression", numeric_cols[-1]

    return None, None
