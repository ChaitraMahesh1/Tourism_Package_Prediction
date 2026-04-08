# for data manipulation
import pandas as pd

# preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# model training
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# model serialization
import joblib

# hugging face
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# mlflow
import mlflow

import os

# ---------------- MLflow Setup ----------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("mlops-training-experiment")

# ---------------- Hugging Face Auth ----------------
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

api = HfApi()

# ---------------- Data Paths ----------------
Xtrain_path = "hf://datasets/chaitram/tourism-package-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/chaitram/tourism-package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/chaitram/tourism-package-prediction/ytrain.csv"
ytest_path = "hf://datasets/chaitram/tourism-package-prediction/ytest.csv"

# ---------------- Load Data ----------------
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

# ---------------- Feature Separation ----------------
categorical_features = Xtrain.select_dtypes(include=['object']).columns.tolist()
numeric_features = Xtrain.select_dtypes(exclude=['object']).columns.tolist()

# ---------------- Class Imbalance Handling ----------------
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# ---------------- Preprocessing ----------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# ---------------- Model ----------------
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42
)

# ---------------- Pipeline ----------------
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ---------------- Hyperparameter Grid ----------------
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# ---------------- Training with MLflow ----------------
with mlflow.start_run():

    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_test_score", results['mean_test_score'][i])
            mlflow.log_metric("std_test_score", results['std_test_score'][i])

    # Log best params
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Threshold tuning
    threshold = 0.45

    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1": test_report['1']['f1-score']
    })

    # ---------------- Save Model ----------------
    model_path = "best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved locally at: {model_path}")

    # ---------------- Hugging Face Upload ----------------
    repo_id = "chaitram/tourism-package-prediction"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Creating repo '{repo_id}'...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type
    )
