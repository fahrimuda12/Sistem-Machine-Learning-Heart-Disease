import requests
import time
import pandas as pd
import joblib
import json

# Fungsi untuk melakukan transformasi data menggunakan pipeline
def preprocess_data(data, pipeline_path, columns):
   pipeline = joblib.load(pipeline_path)
   df = pd.DataFrame([data], columns=columns)
   transformed = pipeline.transform(df)
   return transformed

# Fungsi dummy inference (ganti dengan fungsi inference Anda jika ada)
def inference(transformed_data):
   # Mengirim data ke endpoint model dengan format MLflow 2.x
   payload = {
      "dataframe_records": transformed_data
   }
   response = requests.post("http://localhost:5004/invocations", json=payload)

   # labelling hasil predict
   if response.status_code != 200:
      raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
   
   response_data = response.json()
   
   predictions = response_data.get('predictions', [])
   if not predictions:
      raise Exception("No predictions found in the response")
   
   if predictions[0] == 1:
      print("Prediksi: Terkena Penyakit jantung")
   else:
      print("Prediksi: Tidak terkena Penyakit jantung")
   

   return response

# Data baru
# new_data = [
#    55,                # Age (int)
#    'F',               # Sex (object)
#    'NAP',             # ChestPainType (object)
#    135,               # RestingBP (int)
#    250,               # Cholesterol (int)
#    0,                 # FastingBS (int)
#    'Normal',          # RestingECG (object)
#    150,               # MaxHR (int)
#    'N',               # ExerciseAngina (object)
#    1.0,               # Oldpeak (float)
#    'Flat'             # ST_Slope (object)
# ]

new_data = [
   49,'F','NAP',160,180,0,'Normal',156,'N',1,'Flat'
]

pipeline_path = 'preprocessor_pipeline.joblib'
col = pd.read_csv('data.csv')
columns = col.columns.tolist()

# Preprocessing
transformed_data = preprocess_data(new_data, pipeline_path, columns)

# Ubah hasil transformasi menjadi dict sesuai format yang diinginkan
transformed_dict = dict(zip(columns, transformed_data[0]))
transformed_data = [transformed_dict]

try:
   response = inference(transformed_data)
   print(f"Status: {response.status_code} - Response: {response.text}")
except Exception as e:
   print(f"Request failed: {e}")
