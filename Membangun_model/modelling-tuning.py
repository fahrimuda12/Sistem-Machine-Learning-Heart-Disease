import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from dagshub import dagshub_logger

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Inisialisasi DagsHub
mlflow.set_tracking_uri("https://dagshub.com/fahrimuda12/heart-disease.mlflow")
mlflow.set_experiment("Online Training Heart Disease Failure")

# Load dataset
data = pd.read_csv('./dataset/failure_heart_preprocessing.csv')

# Preprocessing
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"].astype(int)

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'penalty': ['l2', 'elasticnet'],
    'loss': ['log_loss'],
    'learning_rate': ['adaptive'],
    'eta0': [0.01, 0.1]
}

grid = GridSearchCV(
    SGDClassifier(max_iter=1000, random_state=42),
    param_grid,
    cv=3,
    scoring='f1_macro',
    verbose=1,
    n_jobs=-1
)

# Mulai MLflow run
with mlflow.start_run():

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Prediksi
    y_pred = best_model.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    train_test_ratio = len(X_train) / len(X_test)

    # Logging parameter & metric manual
    mlflow.log_params(grid.best_params_)
    mlflow.log_param("model_type", "SGDClassifier + GridSearchCV")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("specificity", specificity)
    mlflow.log_metric("train_test_ratio", train_test_ratio)

    # Log model
    mlflow.sklearn.log_model(best_model, "tuned_model")

    # Log ke DagsHub
    with dagshub_logger() as logger:
        logger.log_hyperparams(grid.best_params_)
        logger.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "specificity": specificity
        })

    # Simpan model lokal
    joblib.dump(best_model, "best_model.pkl")
