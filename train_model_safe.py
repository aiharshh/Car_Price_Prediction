#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production-Safe Advanced Car Price Prediction Training System
Unicode-safe version for deployment on Render.com

This module provides a comprehensive machine learning pipeline for car price prediction
with advanced feature engineering, multiple algorithms, and production-ready deployment.
"""

import os
import sys
import warnings

# Set environment variables for deployment safety
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Suppress warnings
warnings.filterwarnings('ignore')

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load
import json
from datetime import datetime
import time

class ProductionCarPricePrediction:
    """
    Production-safe advanced car price prediction system
    Optimized for deployment environments with encoding safety
    """
    
    def __init__(self, data_path='used_cars_filled.csv'):
        """Initialize the prediction system"""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.feature_columns = None
        self.model_metrics = {}
        
        # Define models for training
        self.model_configs = {
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [6, 8],
                    'learning_rate': [0.1, 0.15]
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15],
                    'min_samples_split': [2, 5]
                }
            },
            'ExtraTrees': {
                'model': ExtraTreesRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15]
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [100, 1000],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf']
                }
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0]
                }
            }
        }
    
    def load_and_explore_data(self):
        """Load and explore the dataset"""
        print("Loading dataset...")
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully. Shape: {self.data.shape}")
            
            # Basic data info
            print("\nDataset Info:")
            print(f"- Rows: {self.data.shape[0]}")
            print(f"- Columns: {self.data.shape[1]}")
            print(f"- Missing values: {self.data.isnull().sum().sum()}")
            
            # Statistical summary
            print("\nStatistical Summary:")
            print(self.data.describe())
            
            return True
            
        except FileNotFoundError:
            print(f"ERROR: Dataset file '{self.data_path}' not found!")
            return False
        except Exception as e:
            print(f"ERROR loading dataset: {str(e)}")
            return False
    
    def feature_engineering(self):
        """Enhanced feature engineering with deployment safety"""
        print("\nPerforming feature engineering...")
        
        try:
            # Create engineered features
            self.data['power_to_weight'] = self.data['max_power(bhp)'] / (self.data['engine(cc)'] / 1000)
            self.data['efficiency_ratio'] = self.data['mileage(kmpl)'] / (self.data['engine(cc)'] / 1000)
            self.data['torque_per_cc'] = self.data['torque(Nm)'] / self.data['engine(cc)']
            self.data['age_factor'] = np.log1p(self.data['kms_driven'])
            self.data['power_efficiency'] = self.data['max_power(bhp)'] / self.data['mileage(kmpl)']
            
            # Handle infinite values
            self.data = self.data.replace([np.inf, -np.inf], np.nan)
            self.data = self.data.fillna(self.data.median())
            
            # Categorical feature engineering
            def categorize_mileage(mileage):
                if mileage <= 10:
                    return 'Low'
                elif mileage <= 15:
                    return 'Medium'
                elif mileage <= 20:
                    return 'High'
                else:
                    return 'Very High'
            
            def categorize_engine(engine):
                if engine <= 1000:
                    return 'Small'
                elif engine <= 1500:
                    return 'Medium'
                elif engine <= 2000:
                    return 'Large'
                else:
                    return 'Very Large'
            
            # Create categorical features
            self.data['mileage_category'] = self.data['mileage(kmpl)'].apply(categorize_mileage)
            self.data['engine_category'] = self.data['engine(cc)'].apply(categorize_engine)
            
            # One-hot encoding
            mileage_dummies = pd.get_dummies(self.data['mileage_category'], prefix='mileage_category')
            engine_dummies = pd.get_dummies(self.data['engine_category'], prefix='engine_category')
            
            # Combine all features
            self.data = pd.concat([self.data, mileage_dummies, engine_dummies], axis=1)
            
            # Drop categorical columns
            self.data = self.data.drop(['mileage_category', 'engine_category'], axis=1)
            
            print(f"Feature engineering completed. New feature count: {self.data.shape[1] - 1}")
            return True
            
        except Exception as e:
            print(f"ERROR in feature engineering: {str(e)}")
            return False
    
    def prepare_data_for_training(self):
        """Prepare data for model training"""
        print("\nPreparing data for training...")
        
        try:
            # Separate features and target
            if 'price(in lakhs)' in self.data.columns:
                target_col = 'price(in lakhs)'
            else:
                print("ERROR: Target column 'price(in lakhs)' not found!")
                return False
            
            X = self.data.drop(target_col, axis=1)
            y = self.data[target_col]
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            print(f"Features: {X.shape[1]}")
            print(f"Samples: {X.shape[0]}")
            print(f"Target: {target_col}")
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            print("Data preparation completed successfully!")
            print(f"Training set: {self.X_train.shape[0]} samples")
            print(f"Test set: {self.X_test.shape[0]} samples")
            
            return True
            
        except Exception as e:
            print(f"ERROR in data preparation: {str(e)}")
            return False
    
    def train_models(self):
        """Train multiple models with cross-validation"""
        print("\nTraining models...")
        
        results = []
        
        for name, config in self.model_configs.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            try:
                # Perform grid search
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=3, 
                    scoring='r2',
                    n_jobs=-1
                )
                
                # Fit the model
                grid_search.fit(self.X_train_scaled, self.y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, self.X_train_scaled, self.y_train, cv=5, scoring='r2')
                
                # Test predictions
                y_pred = best_model.predict(self.X_test_scaled)
                
                # Calculate metrics
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                training_time = time.time() - start_time
                
                # Store results
                metrics = {
                    'r2_score': r2,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time': training_time,
                    'best_params': grid_search.best_params_
                }
                
                self.models[name] = best_model
                self.model_metrics[name] = metrics
                
                results.append((name, r2, mae))
                
                print(f"  R2 Score: {r2:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                print(f"  Training time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"  ERROR training {name}: {str(e)}")
                continue
        
        # Find best model
        if results:
            best_name = max(results, key=lambda x: x[1])[0]
            self.best_model = self.models[best_name]
            
            print(f"\nBest model: {best_name}")
            print(f"Best R2 score: {results[0][1]:.4f}")
            
            return True
        else:
            print("ERROR: No models trained successfully!")
            return False
    
    def save_models(self):
        """Save trained models and components"""
        print("\nSaving models and components...")
        
        try:
            # Create models directory
            os.makedirs('models', exist_ok=True)
            
            # Save best model
            model_path = 'models/best_car_price_model.joblib'
            dump(self.best_model, model_path)
            print(f"Best model saved: {model_path}")
            
            # Save scaler
            scaler_path = 'models/feature_scaler.joblib'
            dump(self.scaler, scaler_path)
            print(f"Scaler saved: {scaler_path}")
            
            # Save feature columns
            features_path = 'models/feature_columns.joblib'
            dump(self.feature_columns, features_path)
            print(f"Feature columns saved: {features_path}")
            
            # Save metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'model_metrics': self.model_metrics,
                'feature_count': len(self.feature_columns),
                'training_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'best_model': max(self.model_metrics.items(), key=lambda x: x[1]['r2_score'])[0]
            }
            
            metadata_path = 'models/model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved: {metadata_path}")
            
            # Also save to root directory for backward compatibility
            dump(self.best_model, 'car_price_model.joblib')
            print("Model also saved to root directory for compatibility")
            
            return True
            
        except Exception as e:
            print(f"ERROR saving models: {str(e)}")
            return False
    
    def print_final_summary(self):
        """Print final training summary"""
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        if self.model_metrics:
            best_model_name = max(self.model_metrics.items(), key=lambda x: x[1]['r2_score'])[0]
            best_metrics = self.model_metrics[best_model_name]
            
            print(f"Best Model: {best_model_name}")
            print(f"R2 Score: {best_metrics['r2_score']:.4f}")
            print(f"MAE: {best_metrics['mae']:.4f}")
            print(f"RMSE: {best_metrics['rmse']:.4f}")
            print(f"Cross-validation: {best_metrics['cv_mean']:.4f} (+/- {best_metrics['cv_std']:.4f})")
            
            print(f"\nDataset Info:")
            print(f"Total samples: {len(self.X_train) + len(self.X_test)}")
            print(f"Features: {len(self.feature_columns)}")
            print(f"Training samples: {len(self.X_train)}")
            print(f"Test samples: {len(self.X_test)}")
        
        print("\nFiles created:")
        print("- models/best_car_price_model.joblib")
        print("- models/feature_scaler.joblib")
        print("- models/feature_columns.joblib")
        print("- models/model_metadata.json")
        print("- car_price_model.joblib (compatibility)")
        
        print("\nReady for deployment!")

def main():
    """Main function for training pipeline"""
    print("Starting Advanced Car Price Prediction Training System...")
    print("Production-safe version for deployment")
    
    # Initialize the system
    predictor = ProductionCarPricePrediction()
    
    # Execute training pipeline
    steps = [
        ("Loading Data", predictor.load_and_explore_data),
        ("Feature Engineering", predictor.feature_engineering),
        ("Data Preparation", predictor.prepare_data_for_training),
        ("Model Training", predictor.train_models),
        ("Saving Models", predictor.save_models)
    ]
    
    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        if not step_func():
            print(f"FATAL ERROR: {step_name} failed!")
            return False
    
    # Print summary
    predictor.print_final_summary()
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nTraining pipeline completed successfully!")
            sys.exit(0)
        else:
            print("\nTraining pipeline failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)
