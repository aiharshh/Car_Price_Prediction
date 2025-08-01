#!/usr/bin/env python3
"""
Render deployment startup script
This ensures the model is trained and ready before starting the Flask app
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_exists():
    """Check if trained model exists"""
    model_paths = [
        'car_price_model.joblib',
        'models/best_car_price_model.joblib'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            logger.info(f"Found trained model at: {path}")
            return True
    
    return False

def train_model():
    """Train the model if it doesn't exist"""
    try:
        logger.info("Training model...")
        result = subprocess.run([sys.executable, 'train_model.py'], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info("Model training completed successfully")
            return True
        else:
            logger.error(f"Model training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Model training timed out")
        return False
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False

def main():
    """Main startup function for Render"""
    logger.info("Starting Render deployment setup...")
    
    # Check if model exists, train if not
    if not check_model_exists():
        logger.info("No trained model found, training now...")
        if not train_model():
            logger.error("Failed to train model, exiting...")
            sys.exit(1)
    
    # Import and start the Flask app
    try:
        from app import app
        logger.info("Flask app imported successfully")
        
        # Get port from environment (Render sets this)
        port = int(os.environ.get("PORT", 5000))
        logger.info(f"Starting server on port {port}")
        
        # Start the app
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
