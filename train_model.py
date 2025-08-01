# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import (mean_absolute_error, r2_score, mean_squared_error, 
                           mean_absolute_percentage_error, explained_variance_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load
import warnings
import time
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
import os

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedCarPricePrediction:
    """
    Advanced Car Price Prediction System with multiple models and comprehensive evaluation
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.feature_importance = {}
        self.evaluation_results = {}
        
    def load_and_explore_data(self) -> pd.DataFrame:
        """Load and perform comprehensive data exploration"""
        logger.info("Loading and exploring data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        
        print("=" * 60)
        print("🚗 ADVANCED CAR PRICE PREDICTION SYSTEM")
        print("=" * 60)
        
        print(f"\n📊 Dataset Shape: {self.data.shape}")
        print(f"📁 Total Features: {self.data.shape[1] - 1}")
        print(f"📈 Total Samples: {self.data.shape[0]}")
        
        # Basic info
        print("\n" + "="*50)
        print("📋 DATASET OVERVIEW")
        print("="*50)
        print(self.data.head(10))
        
        print("\n" + "="*50)
        print("📊 STATISTICAL SUMMARY")
        print("="*50)
        print(self.data.describe())
        
        # Data quality checks
        print("\n" + "="*50)
        print("🔍 DATA QUALITY ANALYSIS")
        print("="*50)
        print(f"Missing Values:\n{self.data.isnull().sum()}")
        print(f"\nDuplicate Rows: {self.data.duplicated().sum()}")
        print(f"Data Types:\n{self.data.dtypes}")
        
        # Statistical tests and distributions
        self._analyze_distributions()
        
        return self.data
    
    def _analyze_distributions(self):
        """Analyze feature distributions and relationships"""
        print("\n" + "="*50)
        print("📈 DISTRIBUTION ANALYSIS")
        print("="*50)
        
        # Create comprehensive visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Distributions and Relationships', fontsize=16, fontweight='bold')
        
        # Distribution plots
        features = ['kms_driven', 'mileage(kmpl)', 'engine(cc)', 'max_power(bhp)', 'torque(Nm)']
        for i, feature in enumerate(features):
            ax = axes[i//3, i%3]
            self.data[feature].hist(bins=30, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'{feature} Distribution', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Price distribution
        ax = axes[1, 2]
        self.data['price(in lakhs)'].hist(bins=30, ax=ax, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.set_title('Price Distribution (Target)', fontweight='bold')
        ax.set_xlabel('Price (in lakhs)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Correlation analysis
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.data.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.3f', 
                   cmap='RdYlBu_r', center=0, square=True, vmin=-1, vmax=1,
                   cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # Feature relationships with target
        self._plot_feature_relationships()
    
    def _plot_feature_relationships(self):
        """Plot relationships between features and target variable"""
        features = ['kms_driven', 'mileage(kmpl)', 'engine(cc)', 'max_power(bhp)', 'torque(Nm)']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature vs Price Relationships', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(features):
            ax = axes[i//3, i%3]
            ax.scatter(self.data[feature], self.data['price(in lakhs)'], 
                      alpha=0.6, color='darkblue', s=20)
            ax.set_xlabel(feature, fontweight='bold')
            ax.set_ylabel('Price (in lakhs)', fontweight='bold')
            ax.set_title(f'{feature} vs Price', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(self.data[feature], self.data['price(in lakhs)'], 1)
            p = np.poly1d(z)
            ax.plot(self.data[feature], p(self.data[feature]), "r--", alpha=0.8, linewidth=2)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.show()
    
    def feature_engineering(self):
        """Advanced feature engineering"""
        logger.info("Performing feature engineering...")
        
        print("\n" + "="*50)
        print("🔧 FEATURE ENGINEERING") 
        print("="*50)
        
        # Create new features
        self.data['power_to_weight'] = self.data['max_power(bhp)'] / (self.data['engine(cc)'] / 1000)
        self.data['efficiency_ratio'] = self.data['mileage(kmpl)'] / (self.data['engine(cc)'] / 1000)
        self.data['torque_per_cc'] = self.data['torque(Nm)'] / self.data['engine(cc)']
        self.data['age_factor'] = np.log1p(self.data['kms_driven'])  # Log transform for better distribution
        self.data['power_efficiency'] = self.data['max_power(bhp)'] / self.data['mileage(kmpl)']
        
        # Binning continuous variables
        self.data['mileage_category'] = pd.cut(self.data['mileage(kmpl)'], 
                                              bins=[0, 10, 15, 20, float('inf')],
                                              labels=['Low', 'Medium', 'High', 'Very High'])
        
        self.data['engine_category'] = pd.cut(self.data['engine(cc)'],
                                            bins=[0, 1000, 1500, 2000, float('inf')],
                                            labels=['Small', 'Medium', 'Large', 'Very Large'])
        
        # One-hot encode categorical features
        categorical_features = ['mileage_category', 'engine_category']
        self.data = pd.get_dummies(self.data, columns=categorical_features, prefix=categorical_features)
        
        print(f"✅ New features created. Total features: {self.data.shape[1] - 1}")
        print("New engineered features:")
        for feature in ['power_to_weight', 'efficiency_ratio', 'torque_per_cc', 'age_factor', 'power_efficiency']:
            print(f"  • {feature}")
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        """Prepare data for training with advanced preprocessing"""
        logger.info("Preparing data for training...")
        
        # Separate features and target
        X = self.data.drop('price(in lakhs)', axis=1)
        y = self.data['price(in lakhs)']
        
        print(f"\n📊 Final dataset shape: {X.shape}")
        print(f"🎯 Target variable: price(in lakhs)")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Feature scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data split completed:")
        print(f"  • Training set: {self.X_train.shape[0]} samples")
        print(f"  • Test set: {self.X_test.shape[0]} samples")
        print(f"  • Features scaled using StandardScaler")
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize multiple regression models with optimized parameters"""
        logger.info("Initializing models...")
        
        self.models = {
            # Linear Models
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            
            # Tree-based Models
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            
            # Other Models
            'Support Vector Regression': SVR(kernel='rbf'),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        print(f"\n🤖 Initialized {len(self.models)} different models:")
        for name in self.models.keys():
            print(f"  • {name}")
        
        return self.models
    
    def cross_validate_models(self, cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation on all models"""
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        print(f"\n🔄 {cv_folds}-FOLD CROSS-VALIDATION")
        print("="*60)
        
        cv_results = {}
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            start_time = time.time()
            
            # Use scaled data for models that benefit from it
            if name in ['Support Vector Regression', 'Neural Network', 'K-Nearest Neighbors']:
                X_data = self.X_train_scaled
            else:
                X_data = self.X_train
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_data, self.y_train, 
                                      cv=kf, scoring='r2', n_jobs=-1)
            
            mae_scores = -cross_val_score(model, X_data, self.y_train,
                                        cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
            
            rmse_scores = np.sqrt(-cross_val_score(model, X_data, self.y_train,
                                                 cv=kf, scoring='neg_mean_squared_error', n_jobs=-1))
            
            end_time = time.time()
            
            cv_results[name] = {
                'R2_mean': cv_scores.mean(),
                'R2_std': cv_scores.std(),
                'MAE_mean': mae_scores.mean(),
                'MAE_std': mae_scores.std(),
                'RMSE_mean': rmse_scores.mean(),
                'RMSE_std': rmse_scores.std(),
                'Training_time': end_time - start_time
            }
            
            print(f"  ✅ R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f}) | Time: {end_time-start_time:.2f}s")
        
        # Display results table
        self._display_cv_results(cv_results)
        return cv_results
    
    def _display_cv_results(self, cv_results: Dict[str, Dict[str, float]]):
        """Display cross-validation results in a formatted table"""
        print(f"\n📊 CROSS-VALIDATION RESULTS SUMMARY")
        print("="*80)
        print(f"{'Model':<25} {'R² Score':<15} {'MAE':<15} {'RMSE':<15} {'Time (s)':<10}")
        print("-"*80)
        
        # Sort by R² score
        sorted_results = sorted(cv_results.items(), key=lambda x: x[1]['R2_mean'], reverse=True)
        
        for name, results in sorted_results:
            print(f"{name:<25} {results['R2_mean']:.4f}±{results['R2_std']:.3f}   "
                  f"{results['MAE_mean']:.4f}±{results['MAE_std']:.3f}   "
                  f"{results['RMSE_mean']:.4f}±{results['RMSE_std']:.3f}   "
                  f"{results['Training_time']:.2f}")
    
    def hyperparameter_tuning(self) -> Dict[str, Any]:
        """Perform hyperparameter tuning for top models"""
        logger.info("Performing hyperparameter tuning...")
        
        print(f"\n🎛️ HYPERPARAMETER TUNING")
        print("="*50)
        
        # Define parameter grids for top models
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Support Vector Regression': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'epsilon': [0.01, 0.1, 0.2]
            }
        }
        
        tuned_models = {}
        
        for model_name, param_grid in param_grids.items():
            print(f"Tuning {model_name}...")
            
            model = self.models[model_name]
            
            # Use appropriate data
            if model_name == 'Support Vector Regression':
                X_data = self.X_train_scaled
            else:
                X_data = self.X_train
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
            )
            
            start_time = time.time()
            grid_search.fit(X_data, self.y_train)
            end_time = time.time()
            
            tuned_models[model_name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'tuning_time': end_time - start_time
            }
            
            print(f"  ✅ Best R² Score: {grid_search.best_score_:.4f}")
            print(f"  ⚙️ Best Parameters: {grid_search.best_params_}")
            print(f"  ⏱️ Tuning Time: {end_time-start_time:.2f}s")
        
        return tuned_models
    
    def train_and_evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """Train all models and perform comprehensive evaluation"""
        logger.info("Training and evaluating models...")
        
        print(f"\n🏋️ MODEL TRAINING & EVALUATION")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            start_time = time.time()
            
            # Use appropriate data
            if name in ['Support Vector Regression', 'Neural Network', 'K-Nearest Neighbors']:
                X_train_data = self.X_train_scaled
                X_test_data = self.X_test_scaled
            else:
                X_train_data = self.X_train
                X_test_data = self.X_test
            
            # Train model
            model.fit(X_train_data, self.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_data)
            
            end_time = time.time()
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(self.y_test, y_pred)
            metrics['training_time'] = end_time - start_time
            
            self.evaluation_results[name] = metrics
            
            # Track best model
            if metrics['r2_score'] > self.best_score:
                self.best_score = metrics['r2_score']
                self.best_model = (name, model)
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            
            print(f"  ✅ R² Score: {metrics['r2_score']:.4f} | MAE: {metrics['mae']:.4f} | Time: {metrics['training_time']:.2f}s")
        
        # Display comprehensive results
        self._display_evaluation_results()
        
        return self.evaluation_results
    
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2_score': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'explained_variance': explained_variance_score(y_true, y_pred),
            'max_error': np.max(np.abs(y_true - y_pred)),
            'mean_residual': np.mean(y_true - y_pred)
        }
    
    def _display_evaluation_results(self):
        """Display comprehensive evaluation results"""
        print(f"\n📈 COMPREHENSIVE MODEL EVALUATION")
        print("="*100)
        print(f"{'Model':<25} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'MAPE':<8} {'Max Err':<8} {'Time':<8}")
        print("-"*100)
        
        # Sort by R² score
        sorted_results = sorted(self.evaluation_results.items(), 
                              key=lambda x: x[1]['r2_score'], reverse=True)
        
        for name, metrics in sorted_results:
            print(f"{name:<25} {metrics['r2_score']:.4f}   {metrics['mae']:.4f}   "
                  f"{metrics['rmse']:.4f}   {metrics['mape']:.2f}%   "
                  f"{metrics['max_error']:.2f}    {metrics['training_time']:.2f}s")
        
        print(f"\n🏆 BEST MODEL: {self.best_model[0]} (R² Score: {self.best_score:.4f})")
    
    def create_advanced_visualizations(self):
        """Create comprehensive visualizations for model analysis"""
        logger.info("Creating advanced visualizations...")
        
        print(f"\n📊 CREATING ADVANCED VISUALIZATIONS")
        print("="*50)
        
        # Model comparison visualization
        self._plot_model_comparison()
        
        # Best model analysis
        self._plot_best_model_analysis()
        
        # Feature importance analysis
        self._plot_feature_importance()
        
        # Residual analysis
        self._plot_residual_analysis()
    
    def _plot_model_comparison(self):
        """Plot model comparison charts"""
        models = list(self.evaluation_results.keys())
        r2_scores = [self.evaluation_results[model]['r2_score'] for model in models]
        mae_scores = [self.evaluation_results[model]['mae'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # R² Score comparison
        bars1 = ax1.bar(range(len(models)), r2_scores, color='skyblue', alpha=0.8, edgecolor='navy')
        ax1.set_xlabel('Models', fontweight='bold')
        ax1.set_ylabel('R² Score', fontweight='bold')
        ax1.set_title('Model Comparison - R² Scores', fontweight='bold')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # MAE comparison
        bars2 = ax2.bar(range(len(models)), mae_scores, color='lightcoral', alpha=0.8, edgecolor='darkred')
        ax2.set_xlabel('Models', fontweight='bold')
        ax2.set_ylabel('Mean Absolute Error', fontweight='bold')
        ax2.set_title('Model Comparison - MAE Scores', fontweight='bold')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars2, mae_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_best_model_analysis(self):
        """Plot detailed analysis for the best model"""
        best_model_name, best_model = self.best_model
        
        # Use appropriate data for prediction
        if best_model_name in ['Support Vector Regression', 'Neural Network', 'K-Nearest Neighbors']:
            X_test_data = self.X_test_scaled
        else:
            X_test_data = self.X_test
        
        y_pred = best_model.predict(X_test_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Best Model Analysis: {best_model_name}', fontsize=16, fontweight='bold')
        
        # Actual vs Predicted
        ax1.scatter(self.y_test, y_pred, alpha=0.7, color='darkblue', s=30)
        ax1.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Prices (lakhs)', fontweight='bold')
        ax1.set_ylabel('Predicted Prices (lakhs)', fontweight='bold')
        ax1.set_title('Actual vs Predicted Prices', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = self.y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.7, color='green', s=30)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Prices (lakhs)', fontweight='bold')
        ax2.set_ylabel('Residuals', fontweight='bold')
        ax2.set_title('Residual Plot', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax3.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('Residuals', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Residuals Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Error distribution
        abs_errors = np.abs(residuals)
        ax4.hist(abs_errors, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Absolute Error', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Absolute Error Distribution', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed metrics for best model
        metrics = self.evaluation_results[best_model_name]
        print(f"\n🏆 BEST MODEL DETAILED METRICS: {best_model_name}")
        print("="*60)
        for metric, value in metrics.items():
            if metric != 'training_time':
                if metric == 'mape':
                    print(f"{metric.upper():<20}: {value:.2f}%")
                else:
                    print(f"{metric.upper():<20}: {value:.4f}")
            else:
                print(f"{metric.upper():<20}: {value:.2f} seconds")
    
    def _plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        if not self.feature_importance:
            print("No feature importance data available (only for tree-based models)")
            return
        
        # Plot feature importance for all tree-based models
        n_models = len(self.feature_importance)
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(1, min(n_models, 3), figsize=(18, 6))
        if n_models == 1:
            axes = [axes]
        
        feature_names = self.X_train.columns
        
        for i, (model_name, importance) in enumerate(list(self.feature_importance.items())[:3]):
            ax = axes[i] if n_models > 1 else axes[0]
            
            # Get top 10 features
            indices = np.argsort(importance)[-10:]
            ax.barh(range(len(indices)), importance[indices], color='lightgreen', alpha=0.8)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[j] for j in indices])
            ax.set_xlabel('Feature Importance', fontweight='bold')
            ax.set_title(f'{model_name}\nTop 10 Features', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_residual_analysis(self):
        """Create comprehensive residual analysis"""
        best_model_name, best_model = self.best_model
        
        # Use appropriate data
        if best_model_name in ['Support Vector Regression', 'Neural Network', 'K-Nearest Neighbors']:
            X_test_data = self.X_test_scaled
        else:
            X_test_data = self.X_test
        
        y_pred = best_model.predict(X_test_data)
        residuals = self.y_test - y_pred
        
        plt.figure(figsize=(15, 10))
        
        # Create a comprehensive residual analysis plot
        plt.subplot(2, 3, 1)
        plt.scatter(y_pred, residuals, alpha=0.6, color='blue')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Q-Q Plot
        from scipy import stats
        plt.subplot(2, 3, 2)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        plt.grid(True, alpha=0.3)
        
        # Scale-Location plot
        plt.subplot(2, 3, 3)
        plt.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.6, color='green')
        plt.xlabel('Predicted Values')
        plt.ylabel('√|Residuals|')
        plt.title('Scale-Location Plot')
        plt.grid(True, alpha=0.3)
        
        # Residuals vs Features (top 3 important features)
        if best_model_name in self.feature_importance:
            importance = self.feature_importance[best_model_name]
            top_features_idx = np.argsort(importance)[-3:]
            
            for i, feature_idx in enumerate(top_features_idx):
                plt.subplot(2, 3, 4 + i)
                plt.scatter(self.X_test.iloc[:, feature_idx], residuals, alpha=0.6)
                plt.axhline(y=0, color='red', linestyle='--')
                plt.xlabel(self.X_test.columns[feature_idx])
                plt.ylabel('Residuals')
                plt.title(f'Residuals vs {self.X_test.columns[feature_idx]}')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model_and_artifacts(self, model_dir: str = "models"):
        """Save the best model and all artifacts"""
        logger.info("Saving model and artifacts...")
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save best model
        best_model_name, best_model = self.best_model
        model_path = os.path.join(model_dir, 'best_car_price_model.joblib')
        dump(best_model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'feature_scaler.joblib')
        dump(self.scaler, scaler_path)
        
        # Save feature columns
        feature_columns_path = os.path.join(model_dir, 'feature_columns.joblib')
        dump(list(self.X_train.columns), feature_columns_path)
        
        # Save evaluation results
        import json
        results_path = os.path.join(model_dir, 'evaluation_results.json')
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for model, metrics in self.evaluation_results.items():
            serializable_results[model] = {k: float(v) for k, v in metrics.items()}
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save model metadata
        metadata = {
            'best_model': best_model_name,
            'best_score': float(self.best_score),
            'timestamp': datetime.now().isoformat(),
            'total_features': self.X_train.shape[1],
            'training_samples': self.X_train.shape[0],
            'test_samples': self.X_test.shape[0]
        }
        
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 MODEL ARTIFACTS SAVED")
        print("="*50)
        print(f"📁 Directory: {model_dir}")
        print(f"🤖 Best Model: {model_path}")
        print(f"⚖️ Scaler: {scaler_path}")
        print(f"📊 Feature Columns: {feature_columns_path}")
        print(f"📈 Evaluation Results: {results_path}")
        print(f"📋 Metadata: {metadata_path}")
        
        # Also save to the original location for Flask app compatibility
        dump(best_model, 'car_price_model.joblib')
        print(f"🔄 Also saved as 'car_price_model.joblib' for Flask app compatibility")
        
        return model_path

def main():
    """Main execution function"""
    print("🚀 Starting Advanced Car Price Prediction System...")
    
    # Initialize the prediction system
    predictor = AdvancedCarPricePrediction(
        r'C:\Users\Om Gaikwad\OneDrive\Desktop\Harsh\Car_Price_Prediction\used_cars_filled_cleaned.csv'
    )
    
    # Execute the complete pipeline
    try:
        # Load and explore data
        predictor.load_and_explore_data()
        
        # Feature engineering
        predictor.feature_engineering()
        
        # Prepare data
        predictor.prepare_data()
        
        # Initialize models
        predictor.initialize_models()
        
        # Cross-validation
        cv_results = predictor.cross_validate_models()
        
        # Hyperparameter tuning for top models
        tuned_models = predictor.hyperparameter_tuning()
        
        # Train and evaluate all models
        evaluation_results = predictor.train_and_evaluate_models()
        
        # Create advanced visualizations
        predictor.create_advanced_visualizations()
        
        # Save model and artifacts
        model_path = predictor.save_model_and_artifacts()
        
        print(f"\n🎉 ADVANCED CAR PRICE PREDICTION SYSTEM COMPLETED SUCCESSFULLY!")
        print(f"🏆 Best Model: {predictor.best_model[0]} with R² Score: {predictor.best_score:.4f}")
        print(f"💾 Model saved at: {model_path}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
