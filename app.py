from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from joblib import load
import os  # Import os module

app = Flask(__name__)

# Load the trained model
model = load('car_price_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')  # Home page

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    kms_driven = float(request.form['kms_driven'])
    mileage_kmpl = float(request.form['mileage_kmpl'])
    engine_cc = float(request.form['engine_cc'])
    max_power_bhp = float(request.form['max_power_bhp'])
    torque_nm = float(request.form['torque_nm'])

    # Prepare input data as DataFrame
    input_data = pd.DataFrame([[kms_driven, mileage_kmpl, engine_cc, max_power_bhp, torque_nm]],
                               columns=['kms_driven', 'mileage(kmpl)', 'engine(cc)', 'max_power(bhp)', 'torque(Nm)'])

    # Make prediction
    predicted_price = model.predict(input_data)

    # Get the absolute value of the predicted price
    predicted_price = abs(predicted_price[0])

    # Return result to the template
    return render_template('result.html', prediction=predicted_price)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use the port Heroku provides
    app.run(debug=True, host='0.0.0.0', port=port)

