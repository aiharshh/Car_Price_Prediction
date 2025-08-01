from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from joblib import load
import os
import json
import logging
from datetime import datetime
import traceback
from werkzeug.exceptions import RequestEntityTooLarge
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model components
model = None
scaler = None
feature_columns = None
model_metadata = None

def load_model_components():
    """Load all model components and metadata"""
    global model, scaler, feature_columns, model_metadata
    
    try:
        # Try to load from models directory first, then fall back to root directory
        model_paths = ['models/best_car_price_model.joblib', 'car_price_model.joblib']
        scaler_paths = ['models/feature_scaler.joblib']
        feature_paths = ['models/feature_columns.joblib']
        metadata_paths = ['models/model_metadata.json']
        
        # Load model
        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                model = load(path)
                model_loaded = True
                logger.info(f"Model loaded from {path}")
                break
        
        if not model_loaded:
            raise FileNotFoundError("No model file found")
        
        # Load scaler if available
        for path in scaler_paths:
            if os.path.exists(path):
                scaler = load(path)
                logger.info(f"Scaler loaded from {path}")
                break
        
        # Load feature columns if available
        for path in feature_paths:
            if os.path.exists(path):
                feature_columns = load(path)
                logger.info(f"Feature columns loaded from {path}")
                break
        
        # Load metadata if available
        for path in metadata_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    model_metadata = json.load(f)
                logger.info(f"Metadata loaded from {path}")
                break
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        return False

def validate_input_data(data):
    """Validate input data for prediction"""
    required_fields = ['kms_driven', 'mileage_kmpl', 'engine_cc', 'max_power_bhp', 'torque_nm']
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        
        try:
            value = float(data[field])
            if value < 0:
                return False, f"Negative value not allowed for {field}"
        except (ValueError, TypeError):
            return False, f"Invalid numeric value for {field}"
    
    # Additional validation
    if float(data['kms_driven']) > 1000000:
        return False, "Kilometers driven seems unrealistic (>1,000,000)"
    
    if float(data['mileage_kmpl']) > 50:
        return False, "Mileage seems unrealistic (>50 kmpl)"
    
    if float(data['engine_cc']) > 10000:
        return False, "Engine capacity seems unrealistic (>10,000cc)"
    
    return True, "Valid"

def prepare_input_features(input_data):
    """Prepare input features with feature engineering"""
    # Basic features
    features = {
        'kms_driven': float(input_data['kms_driven']),
        'mileage(kmpl)': float(input_data['mileage_kmpl']),
        'engine(cc)': float(input_data['engine_cc']),
        'max_power(bhp)': float(input_data['max_power_bhp']),
        'torque(Nm)': float(input_data['torque_nm'])
    }
    
    # Feature engineering (matching training pipeline)
    features['power_to_weight'] = features['max_power(bhp)'] / (features['engine(cc)'] / 1000)
    features['efficiency_ratio'] = features['mileage(kmpl)'] / (features['engine(cc)'] / 1000)
    features['torque_per_cc'] = features['torque(Nm)'] / features['engine(cc)']
    features['age_factor'] = np.log1p(features['kms_driven'])
    features['power_efficiency'] = features['max_power(bhp)'] / features['mileage(kmpl)']
    
    # Categorical features (binning)
    if features['mileage(kmpl)'] <= 10:
        mileage_cat = 'Low'
    elif features['mileage(kmpl)'] <= 15:
        mileage_cat = 'Medium'
    elif features['mileage(kmpl)'] <= 20:
        mileage_cat = 'High'
    else:
        mileage_cat = 'Very High'
    
    if features['engine(cc)'] <= 1000:
        engine_cat = 'Small'
    elif features['engine(cc)'] <= 1500:
        engine_cat = 'Medium'
    elif features['engine(cc)'] <= 2000:
        engine_cat = 'Large'
    else:
        engine_cat = 'Very Large'
    
    # One-hot encoding for categorical features
    mileage_categories = ['Low', 'Medium', 'High', 'Very High']
    engine_categories = ['Small', 'Medium', 'Large', 'Very Large']
    
    for cat in mileage_categories:
        features[f'mileage_category_{cat}'] = 1 if mileage_cat == cat else 0
    
    for cat in engine_categories:
        features[f'engine_category_{cat}'] = 1 if engine_cat == cat else 0
    
    return features

# Load model components on startup
if not load_model_components():
    logger.error("Failed to load model components. Please ensure model files exist.")

@app.route('/')
def home():
    """Home page with enhanced interface"""
    return render_template('index.html', model_metadata=model_metadata)

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint with comprehensive validation"""
    try:
        # Validate input data
        is_valid, message = validate_input_data(request.form)
        if not is_valid:
            return render_template('result.html', 
                                 error=message,
                                 prediction=None)
        
        # Prepare features
        features = prepare_input_features(request.form)
        
        # Create DataFrame with all features
        if feature_columns:
            # Ensure all required features are present
            input_df = pd.DataFrame([features])
            missing_cols = set(feature_columns) - set(input_df.columns)
            for col in missing_cols:
                input_df[col] = 0
            input_df = input_df[feature_columns]
        else:
            # Fallback to basic features
            basic_features = ['kms_driven', 'mileage(kmpl)', 'engine(cc)', 'max_power(bhp)', 'torque(Nm)']
            input_df = pd.DataFrame([[features[col] for col in basic_features]], 
                                  columns=basic_features)
        
        # Scale features if scaler is available
        if scaler:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
        else:
            prediction = model.predict(input_df)
        
        # Get the predicted price
        predicted_price = abs(prediction[0])
        
        # Prepare additional information
        input_summary = {
            'kms_driven': features['kms_driven'],
            'mileage_kmpl': features['mileage(kmpl)'],
            'engine_cc': features['engine(cc)'],
            'max_power_bhp': features['max_power(bhp)'],
            'torque_nm': features['torque(Nm)']
        }
        
        # Calculate confidence interval (simple approach)
        confidence_range = predicted_price * 0.1  # ±10% confidence interval
        
        return render_template('result.html', 
                             prediction=round(predicted_price, 2),
                             confidence_lower=round(predicted_price - confidence_range, 2),
                             confidence_upper=round(predicted_price + confidence_range, 2),
                             input_summary=input_summary,
                             model_info=model_metadata,
                             error=None)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('result.html', 
                             error=f"Prediction failed: {str(e)}",
                             prediction=None)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate input
        is_valid, message = validate_input_data(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Prepare features
        features = prepare_input_features(data)
        
        # Create DataFrame
        if feature_columns:
            input_df = pd.DataFrame([features])
            missing_cols = set(feature_columns) - set(input_df.columns)
            for col in missing_cols:
                input_df[col] = 0
            input_df = input_df[feature_columns]
        else:
            basic_features = ['kms_driven', 'mileage(kmpl)', 'engine(cc)', 'max_power(bhp)', 'torque(Nm)']
            input_df = pd.DataFrame([[features[col] for col in basic_features]], 
                                  columns=basic_features)
        
        # Make prediction
        if scaler:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
        else:
            prediction = model.predict(input_df)
        
        predicted_price = abs(prediction[0])
        confidence_range = predicted_price * 0.1
        
        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'confidence_interval': {
                'lower': round(predicted_price - confidence_range, 2),
                'upper': round(predicted_price + confidence_range, 2)
            },
            'input_features': features,
            'model_info': model_metadata,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'predictions' not in data:
            return jsonify({'error': 'No prediction data provided'}), 400
        
        results = []
        errors = []
        
        for i, item in enumerate(data['predictions']):
            try:
                # Validate each item
                is_valid, message = validate_input_data(item)
                if not is_valid:
                    errors.append({'index': i, 'error': message})
                    continue
                
                # Prepare features
                features = prepare_input_features(item)
                
                # Create DataFrame
                if feature_columns:
                    input_df = pd.DataFrame([features])
                    missing_cols = set(feature_columns) - set(input_df.columns)
                    for col in missing_cols:
                        input_df[col] = 0
                    input_df = input_df[feature_columns]
                else:
                    basic_features = ['kms_driven', 'mileage(kmpl)', 'engine(cc)', 'max_power(bhp)', 'torque(Nm)']
                    input_df = pd.DataFrame([[features[col] for col in basic_features]], 
                                          columns=basic_features)
                
                # Make prediction
                if scaler:
                    input_scaled = scaler.transform(input_df)
                    prediction = model.predict(input_scaled)
                else:
                    prediction = model.predict(input_df)
                
                predicted_price = abs(prediction[0])
                confidence_range = predicted_price * 0.1
                
                results.append({
                    'index': i,
                    'predicted_price': round(predicted_price, 2),
                    'confidence_interval': {
                        'lower': round(predicted_price - confidence_range, 2),
                        'upper': round(predicted_price + confidence_range, 2)
                    }
                })
                
            except Exception as e:
                errors.append({'index': i, 'error': str(e)})
        
        return jsonify({
            'results': results,
            'errors': errors,
            'total_processed': len(results),
            'total_errors': len(errors),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    try:
        info = {
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'feature_columns_loaded': feature_columns is not None,
            'metadata': model_metadata,
            'total_features': len(feature_columns) if feature_columns else None,
            'server_time': datetime.now().isoformat()
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Basic health check
        status = {
            'status': 'healthy',
            'model_loaded': model is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        if model is None:
            status['status'] = 'unhealthy'
            status['error'] = 'Model not loaded'
            return jsonify(status), 503
        
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

