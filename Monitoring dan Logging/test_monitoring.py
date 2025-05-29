import requests
import json
import time
import logging
 
# Konfigurasi logging
logging.basicConfig(filename="api_model_logs.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
 
# Endpoint API model
API_URL = "http://127.0.0.1:5004/invocations"
 
# Contoh input untuk model (ubah sesuai dengan kebutuhan model)
input_data = {"dataframe_split": {"columns": ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"], "data": [[float("nan"), 1, 2, -0.7139208814083184, -1.787420598302848, 0.0, 1, -2.235373456922538, 1, -0.851276473628276, 1], [0.2640268035163891, 0, 0, 0.1103816222175645, 2.124122973867816, 0.0, 0, 0.5185586431894689, 1, 0.9913599439721692, 1]]}}
 
# Konversi data ke JSON
headers = {"Content-Type": "application/json"}
payload = json.dumps(input_data)
 
# Mulai mencatat waktu eksekusi
start_time = time.time()
 
try:
    # Kirim request ke API
    response = requests.post(API_URL, headers=headers, data=payload)
   
    # Hitung response time
    response_time = time.time() - start_time
 
    if response.status_code == 200:
        prediction = response.json()  # Ambil hasil prediksi
 
        # Logging hasil request
        logging.info(f"Request: {input_data}, Response: {prediction}, Response Time: {response_time:.4f} sec")
 
        print(f"Prediction: {prediction}")
        print(f"Response Time: {response_time:.4f} sec")
    else:
        logging.error(f"Error {response.status_code}: {response.text}")
        print(f"Error {response.status_code}: {response.text}")
 
except Exception as e:
    logging.error(f"Exception: {str(e)}")
    print(f"Exception: {str(e)}")