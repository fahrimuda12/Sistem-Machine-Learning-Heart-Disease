# Base image
FROM python:3.11.12-slim-buster

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install MLflow (pastikan versi sesuai dengan proyek kamu)
RUN pip install mlflow

# Install DagsHub logger
RUN pip install dagshub

# Salin seluruh project ke dalam container
COPY . .

# Set environment variables untuk DagsHub logging
# Ganti dengan milikmu atau lewat ARG
ENV MLFLOW_TRACKING_URI=https://dagshub.com/<USERNAME>/<REPO>.mlflow
ENV MLFLOW_TRACKING_USERNAME=<USERNAME>
ENV MLFLOW_TRACKING_PASSWORD=<DAGSHUB_TOKEN>

# Default command to run (bisa disesuaikan)
CMD ["python", "modelling.py"]
