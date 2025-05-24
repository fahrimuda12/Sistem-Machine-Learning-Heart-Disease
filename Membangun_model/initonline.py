from sklearn.datasets import load_iris
import pandas as pd
import mlflow
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from joblib import load
from joblib import dump

# Memuat dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Shuffle dataset untuk simulasi data streaming
data = shuffle(data, random_state=42)

# Split data menjadi batch kecil untuk simulasi data streaming
batch_size = 60  # Ukuran batch
batches = [data.iloc[i:i + batch_size] for i in range(0, len(data), batch_size)]

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Online Training Iris")

# Nama eksperimen yang ingin dicari
experiment_name = "Online Training Iris"

# Mendapatkan ID eksperimen
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Mendapatkan semua runs dalam eksperimen dan memilih run terbaru
runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"])
latest_run_id = runs.iloc[0]["run_id"]  # Run ID terbaru

# Mendapatkan path model dari MLflow artifact
artifact_uri = f"runs:/{latest_run_id}/model_artifacts/online_model.joblib"
local_path = mlflow.artifacts.download_artifacts(artifact_uri)

# Memuat model untuk partial_fit
model = load(local_path)

# Kelas target (diperlukan untuk partial_fit)
classes = data['target'].unique()

with mlflow.start_run():
    mlflow.autolog()  # Melakukan logging otomatis

    # Simulasi Online Training
    # Preprocessing data untuk batch
    X_batch = batches[0].drop(columns=['target'])
    y_batch = batches[0]['target']

        
    # Update model dengan batch berikutnya
    model.partial_fit(X_batch, y_batch)

    # Log metrik setelah setiap batch
    batch_accuracy = model.score(X_batch, y_batch)
    mlflow.log_metric("batch_accuracy", batch_accuracy)
    # Simpan model ke file lokal
    dump(model, "online_model.joblib")

    # Log file model sebagai artefak ke MLflow
    mlflow.log_artifact("online_model.joblib", artifact_path="model_artifacts")
    # Log model setelah selesai online training
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="online_model",
        input_example=X_batch.iloc[:5]
    )