import pandas as pd
import mlflow
import numpy as np
import mlflow.sklearn
import dagshub
import joblib
from dagshub import dagshub_logger
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from mlflow.models.signature import infer_signature



dagshub.init(repo_owner='fahrimuda12', repo_name='heart-disease', mlflow=True)
# Set MLflow Tracking URI
# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_tracking_uri("https://dagshub.com/fahrimuda12/heart-disease.mlflow")

# Create a new MLflow Experiment
mlflow.set_experiment("Online Training Heart Disease Failure")

# Nama eksperimen yang ingin dicari
experiment_name = "Online Training Heart Disease Failure"

# Mendapatkan ID eksperimen
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# # Mendapatkan semua runs dalam eksperimen dan memilih run terbaru
# runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"])
# if runs.empty:
#     raise ValueError("No runs found for the experiment. Please run a training job first.")
# latest_run_id = runs.iloc[0]["run_id"]  # Run ID terbaru

# # Mendapatkan path model dari MLflow artifact
# artifact_uri = f"runs:/{latest_run_id}/model_artifacts/online_model.joblib"
# local_path = mlflow.artifacts.download_artifacts(artifact_uri)

# Memuat dataset 
data = pd.read_csv('./dataset/failure_heart_preprocessing.csv', sep=',')

print(data.info())

# Preprocessing (ubah ini sesuai dataset)
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# Konversi eksplisit ke integer
y = y.astype(int)

print("Unique labels:", y.unique())
print("Label dtype:", y.dtype)

# Dummy encoding if needed
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Online Training Setup ===
batch_size = 60
n_batches = int(np.ceil(len(X_train) / batch_size))
clf = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, max_iter=10000, random_state=42)


# -----------------------------
# STEP 3: MLflow Training
# -----------------------------
# 4. MLflow Logging
with mlflow.start_run():
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X_train))
        X_batch, y_batch = X_train[start:end], y_train[start:end]
        
        if i == 0:
            clf.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
        else:
            clf.partial_fit(X_batch, y_batch)
        
         # Optional: Simpan model sementara tiap batch
        joblib.dump(clf, f"result/model_batch_{i+1}.pkl")

    # Inference setelah semua batch
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    train_test_ratio = len(X_train) / len(X_test)

    # Logging ke MLflow (manual)
    mlflow.log_param("model_type", "SGDClassifier")
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("online_learning", True)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("train_test_ratio", train_test_ratio)
    mlflow.log_metric("n_batches", n_batches)


    # Contoh input
    input_example = X_train.iloc[:2].copy()
    input_example.iloc[0, 0] = np.nan  # Secara manual tambahkan NaN pada satu kolom integer

    # Signature model
    signature = infer_signature(X_train, clf.predict(X_train))

    # Log final model
    mlflow.sklearn.log_model(
        clf, 
        "model_heart_disease_failure",
        input_example=input_example,
        signature=signature
    )

    # DagsHub logger (optional but recommended)
    with dagshub_logger() as logger:
        logger.log_hyperparams({
            "model": "SGDClassifier",
            "batch_size": batch_size,
            "online": True
        })
        logger.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })