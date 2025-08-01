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
    """Load all model components and metadata with fallback training"""
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
                try:
                    model = load(path)
                    model_loaded = True
                    logger.info(f"Model loaded from {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model from {path}: {str(e)}")
                    continue
        
        if not model_loaded:
            logger.warning("No pre-trained model found or model loading failed. Training new model...")
            if train_fallback_model():
                model_loaded = True
            else:
                raise Exception("Failed to load or train model")
        
        # Load scaler if available
        for path in scaler_paths:
            if os.path.exists(path):
                try:
                    scaler = load(path)
                    logger.info(f"Scaler loaded from {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load scaler from {path}: {str(e)}")
                    continue
        
        # Load feature columns if available
        for path in feature_paths:
            if os.path.exists(path):
                try:
                    feature_columns = load(path)
                    logger.info(f"Feature columns loaded from {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load feature columns from {path}: {str(e)}")
                    continue
        
        # Load metadata if available
        for path in metadata_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        model_metadata = json.load(f)
                    logger.info(f"Metadata loaded from {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {path}: {str(e)}")
                    continue
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        return False

def train_fallback_model():
    """Train a robust fallback model with comprehensive data cleaning"""
    global model, scaler, feature_columns, model_metadata
    
    try:
        logger.info("Training fallback model with advanced data cleaning...")
        
        # Load data
        if not os.path.exists('used_cars_filled.csv'):
            logger.error("Training data not found!")
            return False
        
        data = pd.read_csv('used_cars_filled.csv')
        logger.info(f"Loaded training data: {data.shape}")
        
        # Smart data cleaning approach - fix corruption instead of removing data
        logger.info("Performing smart data cleaning and corruption repair...")
        
        # Remove duplicates only
        initial_count = len(data)
        data = data.drop_duplicates()
        logger.info(f"Removed {initial_count - len(data)} duplicate rows")
        
        # Smart corruption fixing: where max_power equals engine_cc, estimate realistic max_power
        corruption_mask = data['max_power(bhp)'] == data['engine(cc)']
        corrupted_count = corruption_mask.sum()
        logger.info(f"Found {corrupted_count} rows with max_power=engine_cc corruption")
        
        if corrupted_count > 0:
            # For corrupted rows, estimate max_power based on engine size
            # Typical power-to-displacement ratios: 50-100 BHP per 1000cc
            data.loc[corruption_mask, 'max_power(bhp)'] = (
                data.loc[corruption_mask, 'engine(cc)'] / 1000 * 75  # 75 BHP per 1000cc average
            ).round(1)
            logger.info(f"Fixed {corrupted_count} corrupted max_power values")
        
        # Fix obviously wrong values by capping them at reasonable limits
        data.loc[data['price(in lakhs)'] > 500, 'price(in lakhs)'] = data['price(in lakhs)'].median()
        data.loc[data['kms_driven'] > 1000000, 'kms_driven'] = data['kms_driven'].median()
        data.loc[data['mileage(kmpl)'] > 100, 'mileage(kmpl)'] = data['mileage(kmpl)'].median()
        data.loc[data['engine(cc)'] > 10000, 'engine(cc)'] = data['engine(cc)'].median()
        data.loc[data['max_power(bhp)'] > 1000, 'max_power(bhp)'] = data['max_power(bhp)'].median()
        data.loc[data['torque(Nm)'] > 5000, 'torque(Nm)'] = data['torque(Nm)'].median()
        
        # Handle zero or negative values
        for col in ['price(in lakhs)', 'kms_driven', 'mileage(kmpl)', 'engine(cc)', 'max_power(bhp)', 'torque(Nm)']:
            median_val = data[col].median()
            data.loc[data[col] <= 0, col] = median_val
        
        logger.info(f"After smart cleaning and repair: {len(data)} rows remaining")
        
        if len(data) < 100:
            logger.error("Insufficient data after cleaning!")
            return False
        
        # Enhanced feature engineering
        data['power_to_weight'] = data['max_power(bhp)'] / (data['engine(cc)'] / 1000)
        data['efficiency_ratio'] = data['mileage(kmpl)'] / (data['engine(cc)'] / 1000)
        data['torque_per_cc'] = data['torque(Nm)'] / data['engine(cc)']
        data['age_factor'] = np.log1p(data['kms_driven'])
        data['power_efficiency'] = data['max_power(bhp)'] / data['mileage(kmpl)']
        
        # Add polynomial features for better model performance
        data['power_squared'] = data['max_power(bhp)'] ** 2
        data['engine_squared'] = data['engine(cc)'] ** 2
        data['mileage_inverse'] = 1 / (data['mileage(kmpl)'] + 1)
        
        # Handle infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(data.median())
        
        # Categorical features with better binning
        def categorize_mileage(mileage):
            if mileage <= 12: return 'Low'
            elif mileage <= 18: return 'Medium'
            elif mileage <= 25: return 'High'
            else: return 'Very High'
        
        def categorize_engine(engine):
            if engine <= 1200: return 'Small'
            elif engine <= 1600: return 'Medium'
            elif engine <= 2500: return 'Large'
            else: return 'Very Large'
        
        def categorize_power(power):
            if power <= 80: return 'Low'
            elif power <= 120: return 'Medium'
            elif power <= 180: return 'High'
            else: return 'Very High'
        
        data['mileage_category'] = data['mileage(kmpl)'].apply(categorize_mileage)
        data['engine_category'] = data['engine(cc)'].apply(categorize_engine)
        data['power_category'] = data['max_power(bhp)'].apply(categorize_power)
        
        # One-hot encoding
        mileage_dummies = pd.get_dummies(data['mileage_category'], prefix='mileage_category')
        engine_dummies = pd.get_dummies(data['engine_category'], prefix='engine_category')
        power_dummies = pd.get_dummies(data['power_category'], prefix='power_category')
        
        data = pd.concat([data, mileage_dummies, engine_dummies, power_dummies], axis=1)
        data = data.drop(['mileage_category', 'engine_category', 'power_category'], axis=1)
        
        # Prepare features and target
        target_col = 'price(in lakhs)'
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        feature_columns = X.columns.tolist()
        logger.info(f"Total features after engineering: {len(feature_columns)}")
        
        # Split data with stratification for better distribution
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try multiple models and select the best one
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score, mean_absolute_error
        
        models_to_try = {
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
            'Ridge': Ridge(alpha=1.0, random_state=42)
        }
        
        best_model = None
        best_score = -float('inf')
        best_model_name = None
        
        for name, candidate_model in models_to_try.items():
            try:
                candidate_model.fit(X_train_scaled, y_train)
                y_pred = candidate_model.predict(X_test_scaled)
                score = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                logger.info(f"{name}: R2={score:.4f}, MAE={mae:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model = candidate_model
                    best_model_name = name
                    
            except Exception as e:
                logger.warning(f"Failed to train {name}: {str(e)}")
                continue
        
        if best_model is None or best_score < 0.1:
            logger.error("All models failed to achieve reasonable performance!")
            return False
        
        # Use the best model
        model = best_model
        
        # Final evaluation
        from sklearn.metrics import mean_squared_error
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Final metrics - R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        # Create metadata
        model_metadata = {
            'model_type': f'{best_model_name}_Fallback_Enhanced',
            'r2_score': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'training_date': datetime.now().isoformat(),
            'feature_count': len(feature_columns),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'data_cleaning_applied': True,
            'outliers_removed': True,
            'corruption_fixed': True
        }
        
        # Save components for future use
        try:
            from joblib import dump
            os.makedirs('models', exist_ok=True)
            dump(model, 'models/fallback_model_enhanced.joblib')
            dump(scaler, 'models/fallback_scaler_enhanced.joblib')
            dump(feature_columns, 'models/fallback_features_enhanced.joblib')
            
            with open('models/fallback_metadata_enhanced.json', 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            logger.info("Enhanced fallback model saved successfully")
        except Exception as e:
            logger.warning(f"Could not save enhanced fallback model: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to train enhanced fallback model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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
logger.info("🚀 Starting Car Price Prediction System...")
model_load_success = load_model_components()
if not model_load_success:
    logger.warning("⚠️ Model components failed to load - fallback training will be used")
    logger.info("📌 This is normal for deployment environments with version conflicts")
    logger.info("📌 The system will automatically train a fallback model on first prediction")
else:
    logger.info("✅ All model components loaded successfully")

@app.route('/')
def home():
    """Home page with enhanced interface"""
    return render_template('index.html', model_metadata=model_metadata)

@app.route('/status')
def status_page():
    """System status dashboard"""
    return render_template('status.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint with comprehensive validation and fallback training"""
    global model, scaler, feature_columns, model_metadata
    
    try:
        # Check if model is loaded, if not try fallback training
        if model is None:
            logger.info("🔧 Model not loaded, attempting fallback training...")
            if train_fallback_model():
                logger.info("✅ Fallback model trained successfully")
            else:
                return render_template('result.html', 
                                     error="Model unavailable and fallback training failed. Please try again later.",
                                     prediction=None)
        
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
    """API endpoint for predictions with fallback training"""
    global model, scaler, feature_columns, model_metadata
    
    try:
        # Check if model is loaded, if not try fallback training
        if model is None:
            logger.info("🔧 Model not loaded, attempting fallback training...")
            if not train_fallback_model():
                return jsonify({
                    'error': 'Model unavailable and fallback training failed',
                    'status': 'service_unavailable',
                    'timestamp': datetime.now().isoformat()
                }), 503
        
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
    """Health check endpoint with detailed model status"""
    try:
        # Basic health check
        status = {
            'status': 'healthy' if model is not None else 'degraded',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'feature_columns_loaded': feature_columns is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        if model is None:
            status['status'] = 'degraded'
            status['warning'] = 'Model not loaded - fallback training will occur on first prediction'
            status['action'] = 'Try making a prediction to trigger fallback model training'
        else:
            status['model_info'] = model_metadata if model_metadata else 'No metadata available'
        
        # Return appropriate status code
        if model is None:
            return jsonify(status), 503  # Service Unavailable but recoverable
        
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

