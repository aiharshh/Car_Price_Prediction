# Render deployment script
# This file tells Render how to build and deploy your application

#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Train the model (if not already trained)
if [ ! -f "car_price_model.joblib" ] && [ ! -f "models/best_car_price_model.joblib" ]; then
    echo "Training model..."
    python train_model_safe.py
fi
