"""
Advanced Model Evaluation Script
Provides comprehensive evaluation metrics and analysis for the car price prediction model
"""

import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.model_selection import learning_curve, validation_curve
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, model_path='car_price_model.joblib', data_path='used_cars_filled_cleaned.csv'):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.data = None
        self.results = {}
    
    def load_components(self):
        """Load model and data"""
        try:
            # Load model
            if os.path.exists('models/best_car_price_model.joblib'):
                self.model = load('models/best_car_price_model.joblib')
                print("✅ Loaded advanced model from models directory")
            else:
                self.model = load(self.model_path)
                print("✅ Loaded basic model")
            
            # Load data
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Loaded dataset: {self.data.shape}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading components: {str(e)}")
            return False
    
    def evaluate_model_performance(self):
        """Comprehensive model performance evaluation"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Prepare data
        X = self.data.drop('price(in lakhs)', axis=1)
        y = self.data['price(in lakhs)']
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        metrics = {
            'MAE': mean_absolute_error(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'R²': r2_score(y, y_pred),
            'MAPE': mean_absolute_percentage_error(y, y_pred) * 100,
            'Explained Variance': explained_variance_score(y, y_pred),
            'Max Error': np.max(np.abs(y - y_pred)),
            'Mean Residual': np.mean(y - y_pred)
        }
        
        # Display metrics
        print(f"{'Metric':<20} {'Value':<15} {'Interpretation'}")
        print("-" * 60)
        print(f"{'MAE':<20} {metrics['MAE']:.4f}      Average error in lakhs")
        print(f"{'RMSE':<20} {metrics['RMSE']:.4f}      Root mean squared error")
        print(f"{'R² Score':<20} {metrics['R²']:.4f}      Coefficient of determination")
        print(f"{'MAPE':<20} {metrics['MAPE']:.2f}%       Mean absolute percentage error")
        print(f"{'Max Error':<20} {metrics['Max Error']:.2f}      Largest prediction error")
        
        self.results['performance_metrics'] = metrics
        
        # Visualizations
        self._create_performance_plots(y, y_pred)
        
        return metrics
    
    def _create_performance_plots(self, y_true, y_pred):
        """Create comprehensive performance visualization plots"""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue', s=30)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Prices (lakhs)')
        axes[0, 0].set_ylabel('Predicted Prices (lakhs)')
        axes[0, 0].set_title('Actual vs Predicted Prices')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² score to the plot
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residuals vs Predicted
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green', s=30)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Predicted Prices (lakhs)')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals Distribution
        axes[0, 2].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Residuals Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Absolute Error Distribution
        abs_errors = np.abs(residuals)
        axes[1, 1].hist(abs_errors, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Absolute Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Error by Price Range
        price_bins = pd.cut(y_true, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        error_by_range = pd.DataFrame({'Price_Range': price_bins, 'Absolute_Error': abs_errors})
        error_by_range.boxplot(column='Absolute_Error', by='Price_Range', ax=axes[1, 2])
        axes[1, 2].set_title('Error Distribution by Price Range')
        axes[1, 2].set_xlabel('Price Range')
        axes[1, 2].set_ylabel('Absolute Error')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_analysis(self):
        """Analyze feature importance if available"""
        print("\n" + "="*60)
        print("🔍 FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        if hasattr(self.model, 'feature_importances_'):
            # Get feature names
            feature_names = self.data.drop('price(in lakhs)', axis=1).columns
            importances = self.model.feature_importances_
            
            # Create DataFrame for easier handling
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            print("-" * 40)
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"{i:2d}. {row['Feature']:<25} {row['Importance']:.4f}")
            
            # Visualization
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)
            
            bars = plt.barh(range(len(top_features)), top_features['Importance'], color='skyblue', alpha=0.8)
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importances', fontweight='bold', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{top_features.iloc[i]["Importance"]:.3f}',
                        va='center', ha='left', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            self.results['feature_importance'] = importance_df.to_dict('records')
            
        else:
            print("❌ Feature importance not available for this model type")
    
    def prediction_confidence_analysis(self, sample_size=100):
        """Analyze prediction confidence with random samples"""
        print("\n" + "="*60)
        print("🎯 PREDICTION CONFIDENCE ANALYSIS")
        print("="*60)
        
        # Take random samples
        sample_data = self.data.sample(n=min(sample_size, len(self.data)), random_state=42)
        X_sample = sample_data.drop('price(in lakhs)', axis=1)
        y_true = sample_data['price(in lakhs)']
        y_pred = self.model.predict(X_sample)
        
        # Calculate confidence metrics
        absolute_errors = np.abs(y_true - y_pred)
        percentage_errors = (absolute_errors / y_true) * 100
        
        confidence_metrics = {
            'Mean Absolute Error': np.mean(absolute_errors),
            'Median Absolute Error': np.median(absolute_errors),
            'Mean Percentage Error': np.mean(percentage_errors),
            'Median Percentage Error': np.median(percentage_errors),
            '90% Predictions within': f"±{np.percentile(absolute_errors, 90):.2f} lakhs",
            '95% Predictions within': f"±{np.percentile(absolute_errors, 95):.2f} lakhs"
        }
        
        print("Confidence Metrics on Sample Data:")
        print("-" * 40)
        for metric, value in confidence_metrics.items():
            if isinstance(value, str):
                print(f"{metric:<25}: {value}")
            else:
                print(f"{metric:<25}: {value:.4f}")
        
        # Visualize confidence intervals
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue', s=40)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        
        # Add confidence bands
        confidence_band = np.percentile(absolute_errors, 90)
        x_line = np.linspace(y_true.min(), y_true.max(), 100)
        plt.fill_between(x_line, x_line - confidence_band, x_line + confidence_band, 
                        alpha=0.2, color='red', label=f'90% Confidence Band (±{confidence_band:.2f})')
        
        plt.xlabel('Actual Prices (lakhs)')
        plt.ylabel('Predicted Prices (lakhs)')
        plt.title('Prediction Confidence Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(percentage_errors, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(np.mean(percentage_errors), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(percentage_errors):.1f}%')
        plt.axvline(np.median(percentage_errors), color='orange', linestyle='--', 
                   label=f'Median: {np.median(percentage_errors):.1f}%')
        plt.xlabel('Percentage Error (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Percentage Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.results['confidence_analysis'] = confidence_metrics
    
    def generate_evaluation_report(self, save_path='model_evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("📄 GENERATING EVALUATION REPORT")
        print("="*60)
        
        # Add metadata
        self.results['evaluation_metadata'] = {
            'evaluation_date': datetime.now().isoformat(),
            'model_path': self.model_path,
            'data_path': self.data_path,
            'dataset_size': len(self.data),
            'model_type': str(type(self.model).__name__)
        }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"✅ Evaluation report saved to: {save_path}")
        
        # Print summary
        if 'performance_metrics' in self.results:
            metrics = self.results['performance_metrics']
            print(f"\n📊 EVALUATION SUMMARY")
            print("-" * 30)
            print(f"Model Accuracy (R²):     {metrics['R²']:.4f}")
            print(f"Average Error (MAE):     {metrics['MAE']:.4f} lakhs")
            print(f"RMSE:                    {metrics['RMSE']:.4f} lakhs")
            print(f"MAPE:                    {metrics['MAPE']:.2f}%")
            
            # Performance rating
            r2_score = metrics['R²']
            if r2_score > 0.9:
                rating = "Excellent ⭐⭐⭐⭐⭐"
            elif r2_score > 0.8:
                rating = "Very Good ⭐⭐⭐⭐"
            elif r2_score > 0.7:
                rating = "Good ⭐⭐⭐"
            elif r2_score > 0.6:
                rating = "Fair ⭐⭐"
            else:
                rating = "Needs Improvement ⭐"
            
            print(f"Model Rating:            {rating}")
        
        return self.results

def main():
    """Main evaluation function"""
    print("🚀 Starting Advanced Model Evaluation...")
    
    evaluator = ModelEvaluator()
    
    if not evaluator.load_components():
        print("❌ Failed to load components. Exiting...")
        return
    
    # Run comprehensive evaluation
    try:
        # Model performance evaluation
        performance_metrics = evaluator.evaluate_model_performance()
        
        # Feature importance analysis
        evaluator.feature_importance_analysis()
        
        # Confidence analysis
        evaluator.prediction_confidence_analysis()
        
        # Generate report
        report = evaluator.generate_evaluation_report()
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
