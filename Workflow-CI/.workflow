name: Retrain ML Model

on:
  push:
    paths:
      - 'MLProject/**'
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v3

    - name: Set up Conda
      uses: conda-incubator/setup-miniconda@v2
      with:
        activate-environment: mlflow-env
        environment-file: MLProject/conda.yaml
        auto-activate-base: false

    - name: Install MLflow
      run: |
        pip install mlflow dagshub

    - name: Run MLflow Project
      run: |
        cd MLProject
        mlflow run . --experiment-name "CI Retraining Heart Disease"
