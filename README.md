# ML Algorithm Finder

A Streamlit app that uploads CSV or Excel data, infers whether the selected target is a classification or regression problem, compares multiple machine learning algorithms, and recommends the best-performing model on a holdout test split.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

You can also double-click `run_app.bat` on Windows after installing the requirements.

## Supported Files

- `.csv`
- `.xlsx`
- `.xlsm`
- `.xls`

## What the App Does

1. Uploads tabular data.
2. Lets you choose the target column.
3. Detects whether the task is classification or regression.
4. Builds preprocessing pipelines for numeric and categorical features.
5. Trains several candidate algorithms.
6. Shows the recommended algorithm with training and test performance.

## Algorithms Evaluated

Classification:

- Most frequent class baseline
- Logistic regression
- Decision tree classifier
- Random forest classifier
- K-nearest neighbors classifier
- Support vector classifier
- Gaussian naive Bayes

Regression:

- Mean value baseline
- Linear regression
- Ridge regression
- Decision tree regressor
- Random forest regressor
- K-nearest neighbors regressor
- Support vector regressor

Classification reports training and test accuracy. Regression reports training and test R2 score, plus mean absolute error for the recommended model.
