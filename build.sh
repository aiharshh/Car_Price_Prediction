#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set Python environment variables for better compatibility
export PYTHONIOENCODING=utf-8
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Train the model (if not already trained)
echo "Checking for existing models..."
if [ ! -f "car_price_model.joblib" ] && [ ! -f "models/best_car_price_model.joblib" ]; then
    echo "No existing model found. Training new model..."
    python train_model_safe.py || {
        echo "Model training failed, but app will use fallback training on startup"
        echo "This is normal for deployment environments with version conflicts"
    }
else
    echo "Existing model found, skipping training"
fi

echo "Build completed successfully"
