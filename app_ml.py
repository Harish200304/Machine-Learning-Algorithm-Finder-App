from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="ML Algorithm Finder",
    layout="wide",
)


@dataclass
class ModelResult:
    name: str
    train_score: float
    test_score: float
    gap: float
    model: Pipeline


def read_dataset(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    if not raw:
        raise ValueError("The uploaded file is empty.")

    if file_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    if file_name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(raw))

    raise ValueError("Please upload a CSV or Excel file.")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [
        str(column).strip() or f"column_{index + 1}"
        for index, column in enumerate(cleaned.columns)
    ]
    return cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")


def guess_target_column(df: pd.DataFrame) -> str:
    common_names = [
        "target",
        "label",
        "class",
        "category",
        "outcome",
        "result",
        "price",
        "sales",
        "value",
        "y",
    ]
    lookup = {column.lower(): column for column in df.columns}
    for name in common_names:
        if name in lookup:
            return lookup[name]
    return df.columns[-1]


def infer_task_type(y: pd.Series) -> str:
    y_without_na = y.dropna()
    if y_without_na.empty:
        raise ValueError("The target column does not contain usable values.")

    numeric_target = pd.to_numeric(y_without_na, errors="coerce")
    numeric_ratio = numeric_target.notna().mean()
    is_numeric = pd.api.types.is_numeric_dtype(y_without_na) or numeric_ratio > 0.95
    unique_count = y_without_na.nunique(dropna=True)
    unique_ratio = unique_count / max(len(y_without_na), 1)

    if is_numeric and unique_count > 12 and unique_ratio > 0.08:
        return "Regression"
    return "Classification"


def make_label_encoder() -> LabelEncoder:
    try:
        return LabelEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return LabelEncoder(handle_unknown="ignore", sparse=False)


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    working = df.dropna(subset=[target_column]).copy()
    if len(working) < 10:
        raise ValueError("Use at least 10 rows with a non-empty target value.")

    X = working.drop(columns=[target_column])
    y = working[target_column]
    if X.shape[1] == 0:
        raise ValueError("The dataset needs at least one feature column besides the target.")

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("label", make_label_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def classification_models(random_state: int) -> list[tuple[str, object]]:
    return [
        ("Logistic Regression", LogisticRegression(max_iter=2000)),
        ("Decision Tree Classifier", DecisionTreeClassifier(random_state=random_state)),
        ("Random Forest Classifier", RandomForestClassifier(n_estimators=150, random_state=random_state)),
        ("K-Nearest Neighbors Classifier", KNeighborsClassifier()),
        ("Support Vector Classifier", SVC()),
        ("Gradient Boosting Classifier", GradientBoostingClassifier(random_state=random_state)),
        ("XGBoost Classifier", xgb.XGBClassifier(random_state=random_state, eval_metric="logloss")),
        ("LightGBM Classifier", lgb.LGBMClassifier(random_state=random_state, verbose=-1)),
    ]


def regression_models(random_state: int) -> list[tuple[str, object]]:
    return [
        ("Linear Regression", LinearRegression()),
    ]


def score_models(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    test_size: float,
    random_state: int,
) -> tuple[list[ModelResult], int, int]:
    if task_type == "Regression":
        y = pd.to_numeric(y, errors="coerce")
        valid_target = y.notna()
        X = X.loc[valid_target]
        y = y.loc[valid_target]
        if len(y) < 10:
            raise ValueError("Regression needs at least 10 rows with a numeric target value.")

    stratify = y if task_type == "Classification" and y.nunique() > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    preprocessor = build_preprocessor(X)
    candidates = (
        classification_models(random_state)
        if task_type == "Classification"
        else regression_models(random_state)
    )
    results: list[ModelResult] = []

    for name, estimator in candidates:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        try:
            pipeline.fit(X_train, y_train)
            train_score = pipeline.score(X_train, y_train)
            test_score = pipeline.score(X_test, y_test)

            results.append(
                ModelResult(
                    name=name,
                    train_score=float(train_score),
                    test_score=float(test_score),
                    gap=abs(float(train_score) - float(test_score)),
                    model=pipeline,
                )
            )
        except Exception as exc:
            st.warning(f"{name} was skipped: {exc}")

    if not results:
        raise ValueError("No algorithms could be trained on this dataset.")

    return sorted(results, key=lambda item: (item.test_score, -item.gap), reverse=True), len(X_train), len(X_test)


def format_score(value: float, task_type: str) -> str:
    if task_type == "Classification":
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"


def result_table(results: list[ModelResult], task_type: str) -> pd.DataFrame:
    metric = "Accuracy" if task_type == "Classification" else "R2 Score"
    return pd.DataFrame(
        {
            "Algorithm": [result.name for result in results],
            f"Training {metric}": [format_score(result.train_score, task_type) for result in results],
            f"Test {metric}": [format_score(result.test_score, task_type) for result in results],
            "Generalization Gap": [
                format_score(result.gap, task_type) if task_type == "Classification" else f"{result.gap:.4f}"
                for result in results
            ],
        }
    )


def regression_error_summary(best: ModelResult, X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> str:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    train_predictions = best.model.predict(X_train)
    test_predictions = best.model.predict(X_test)


def render_sidebar(df: pd.DataFrame) -> tuple[str, float, int]:
    st.sidebar.header("Settings")
    target_guess = guess_target_column(df)
    target_column = st.sidebar.selectbox(
        "Target column",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(target_guess),
    )
    test_size = st.sidebar.slider("Test data size", 0.1, 0.4, 0.25, 0.05)
    random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    return target_column, test_size, int(random_state)


def main() -> None:
    st.title("ML Algorithm Finder")
    st.caption("Upload a CSV or Excel dataset, choose the target column, and compare suitable ML algorithms.")

    uploaded_file = st.file_uploader("Upload CSV or Excel data", type=["csv", "xlsx", "xlsm", "xls"])
    if uploaded_file is None:
        st.info("Upload a dataset to begin.")
        return

    try:
        df = clean_columns(read_dataset(uploaded_file))
        if df.empty or len(df.columns) < 2:
            raise ValueError("The dataset must contain at least two usable columns.")

        target_column, test_size, random_state = render_sidebar(df)
        X, y = split_features_target(df, target_column)
        task_type = infer_task_type(y)
        if task_type == "Regression":
            y = pd.to_numeric(y, errors="coerce")
            valid_target = y.notna()
            X = X.loc[valid_target]
            y = y.loc[valid_target]
            if len(y) < 10:
                raise ValueError("Regression needs at least 10 rows with a numeric target value.")

        st.subheader("Dataset Overview")
        summary_cols = st.columns(5)
        summary_cols[0].metric("Rows", f"{len(df):,}")
        summary_cols[1].metric("Columns", f"{len(df.columns):,}")
        summary_cols[2].metric("Features", f"{X.shape[1]:,}")
        summary_cols[3].metric("Target", target_column)
        summary_cols[4].metric("Task", task_type)

        with st.expander("Preview data", expanded=False):
            st.dataframe(df.head(100), use_container_width=True)

        if st.button("Find Best Algorithm", type="primary"):
            with st.spinner("Training and comparing algorithms..."):
                results, train_rows, test_rows = score_models(X, y, task_type, test_size, random_state)
                best = results[0]

            metric_name = "Accuracy" if task_type == "Classification" else "R2 Score"
            st.subheader("Recommended Algorithm")
            best_cols = st.columns(4)
            best_cols[0].metric("Algorithm", best.name)
            best_cols[1].metric(f"Training {metric_name}", format_score(best.train_score, task_type))
            best_cols[2].metric(f"Test {metric_name}", format_score(best.test_score, task_type))
            best_cols[3].metric("Train/Test Rows", f"{train_rows:,} / {test_rows:,}")

            st.success(
                f"{best.name} is recommended because it produced the strongest test {metric_name.lower()} "
                "among the algorithms that successfully trained on this dataset."
            )

            if task_type == "Regression":
                st.caption(regression_error_summary(best, X, y, test_size, random_state))
                st.caption("Regression uses R2 score. Higher is better; 1.0 is perfect and values below 0 can occur.")

            st.subheader("Algorithm Comparison")
            st.dataframe(result_table(results, task_type), use_container_width=True, hide_index=True)

            st.bar_chart(
                pd.DataFrame(
                    {
                        "Algorithm": [result.name for result in results],
                        f"Test {metric_name}": [result.test_score for result in results],
                    }
                ).set_index("Algorithm")
            )

    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()