import requests
import json

# API endpoint
url = "http://localhost:5000/api/predict"

# Car data to predict
car_data = {
    "kms_driven": 45000,
    "mileage_kmpl": 18.5,
    "engine_cc": 1500,
    "max_power_bhp": 120,
    "torque_nm": 200
}

# Make the request
response = requests.post(url, json=car_data)

if response.status_code == 200:
    result = response.json()
    print(f"Predicted Price: ₹{result['predicted_price']} lakhs")
    print(f"Confidence Range: ₹{result['confidence_interval']['lower']} - ₹{result['confidence_interval']['upper']} lakhs")
else:
    print(f"Error: {response.json()['error']}")
