import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# Inisialisasi MLflow autolog
mlflow.sklearn.autolog()

# Muat dataset
data = pd.read_csv('./dataset/failure_heart_preprocessing.csv', sep=',')

# Preprocessing
X = pd.get_dummies(data.drop("HeartDisease", axis=1))
y = data["HeartDisease"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, max_iter=10000, random_state=42)

with mlflow.start_run():
    clf.fit(X_train, y_train)
    # Semua parameter, metrics, dan model akan otomatis dilog oleh autolog

    # Jika ingin, bisa tambahkan evaluasi manual
    score = clf.score(X_test, y_test)
    print(f"Test accuracy: {score}")
