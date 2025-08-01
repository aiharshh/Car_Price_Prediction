# Advanced Car Price Prediction System

## 🚗 Overview

This is an advanced, production-ready car price prediction system that uses machine learning to estimate car prices based on various features. The system includes multiple algorithms, comprehensive evaluation, and a user-friendly web interface.

## 🌟 Features

### Core Features

- **Multiple ML Algorithms**: Linear Regression, Random Forest, Gradient Boosting, SVM, Neural Networks, and more
- **Advanced Feature Engineering**: Automated feature creation and selection
- **Hyperparameter Tuning**: Automated optimization of model parameters
- **Cross-Validation**: Robust model evaluation with k-fold cross-validation
- **Comprehensive Metrics**: MAE, RMSE, R², MAPE, and more

### Web Interface

- **Modern UI**: Responsive design with Bootstrap and custom styling
- **Real-time Validation**: Input validation with helpful suggestions
- **Confidence Intervals**: Prediction ranges with uncertainty estimates
- **Model Information**: Display of model performance and metadata
- **API Endpoints**: RESTful API for integration

### Production Features

- **Health Monitoring**: Health check endpoints
- **Error Handling**: Comprehensive error handling and logging
- **Batch Predictions**: Process multiple predictions at once
- **Performance Monitoring**: Response time tracking
- **Scalable Architecture**: Ready for cloud deployment

## 📁 Project Structure

```
Car_Price_Prediction/
├── app.py                      # Flask web application
├── train_model.py              # Advanced model training pipeline
├── model_evaluation.py         # Comprehensive model evaluation
├── test_api.py                 # API testing suite
├── requirements.txt            # Python dependencies
├── Procfile                    # Heroku deployment configuration
├── README.md                   # This file
├── used_cars_filled.csv        # Dataset
├── car_price_model.joblib      # Trained model (generated)
├── templates/
│   ├── index.html              # Main web interface
│   └── result.html             # Results display page
└── models/                     # Advanced model artifacts (generated)
    ├── best_car_price_model.joblib
    ├── feature_scaler.joblib
    ├── feature_columns.joblib
    ├── evaluation_results.json
    └── model_metadata.json
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Local Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Car_Price_Prediction
   ```

2. **Create virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**

   ```bash
   python train_model.py
   ```

5. **Run the application**

   ```bash
   python app.py
   ```

6. **Access the application**
   - Open your browser and go to `http://localhost:5000`

## 🚀 Usage

### Web Interface

1. **Navigate to the home page**
2. **Enter car specifications:**
   - Kilometers driven
   - Mileage (km/l)
   - Engine capacity (cc)
   - Maximum power (BHP)
   - Torque (Nm)
3. **Click "Predict Car Price"**
4. **View results** with confidence intervals and model information

### API Usage

#### Single Prediction

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "kms_driven": 45000,
    "mileage_kmpl": 18.5,
    "engine_cc": 1500,
    "max_power_bhp": 120,
    "torque_nm": 200
  }'
```

#### Batch Predictions

```bash
curl -X POST http://localhost:5000/api/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "kms_driven": 25000,
        "mileage_kmpl": 20.0,
        "engine_cc": 1200,
        "max_power_bhp": 85,
        "torque_nm": 150
      },
      {
        "kms_driven": 55000,
        "mileage_kmpl": 15.5,
        "engine_cc": 1800,
        "max_power_bhp": 140,
        "torque_nm": 250
      }
    ]
  }'
```

#### Health Check

```bash
curl http://localhost:5000/health
```

#### Model Information

```bash
curl http://localhost:5000/model_info
```

## 🧪 Testing

### Run Model Evaluation

```bash
python model_evaluation.py
```

### Run API Tests

```bash
python test_api.py
```

### Run API Tests with Custom URL

```bash
python test_api.py --url http://your-deployed-app.com
```

## 📊 Model Performance

The system automatically evaluates multiple algorithms and selects the best performer:

- **Algorithms Tested**: Linear Regression, Ridge, Lasso, Elastic Net, Random Forest, Extra Trees, Gradient Boosting, Decision Tree, SVR, KNN, Neural Network
- **Evaluation Metrics**: R² Score, MAE, RMSE, MAPE, Cross-validation scores
- **Feature Engineering**: Power-to-weight ratio, efficiency metrics, categorical binning
- **Hyperparameter Tuning**: Grid search for optimal parameters

### Sample Performance Metrics

- **R² Score**: 0.85+ (85%+ variance explained)
- **MAE**: <3.5 lakhs average error
- **MAPE**: <15% percentage error
- **Response Time**: <200ms per prediction

## 🌐 Deployment

### Heroku Deployment

1. **Install Heroku CLI**
2. **Login to Heroku**

   ```bash
   heroku login
   ```

3. **Create Heroku app**

   ```bash
   heroku create your-app-name
   ```

4. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy advanced car price prediction"
   git push heroku main
   ```

### Docker Deployment

1. **Create Dockerfile**

   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 5000
   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
   ```

2. **Build and run**
   ```bash
   docker build -t car-price-predictor .
   docker run -p 5000:5000 car-price-predictor
   ```

### AWS/GCP Deployment

The application is ready for cloud deployment with:

- Environment variable support
- Health check endpoints
- Logging configuration
- Error handling
- Scalable architecture

## 🔧 Configuration

### Environment Variables

- `PORT`: Server port (default: 5000)
- `FLASK_DEBUG`: Debug mode (default: False)
- `MODEL_PATH`: Custom model path
- `DATA_PATH`: Custom dataset path

### Model Configuration

Modify `train_model.py` to adjust:

- Cross-validation folds
- Hyperparameter search spaces
- Feature engineering parameters
- Model selection criteria

## 📈 Monitoring

### Health Monitoring

- `/health` endpoint for basic health checks
- `/model_info` endpoint for model status
- Logging configuration for error tracking

### Performance Monitoring

- Response time tracking
- Error rate monitoring
- Model accuracy tracking
- Resource usage monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues

**Model not loading**

- Ensure `train_model.py` has been run successfully
- Check if model files exist in the correct location

**Prediction errors**

- Validate input ranges (negative values not allowed)
- Check for missing required fields
- Ensure realistic values (e.g., mileage < 50 km/l)

**Performance issues**

- Monitor system resources
- Consider model optimization
- Check for memory leaks

### Getting Help

1. Check the logs for error messages
2. Run the test suite to identify issues
3. Review the model evaluation report
4. Open an issue on GitHub with details

## 🔮 Future Enhancements

- **Additional Features**: Brand, model year, fuel type, transmission
- **Deep Learning**: Neural network architectures for better accuracy
- **Time Series**: Price trend analysis and forecasting
- **Mobile App**: React Native or Flutter mobile interface
- **Real-time Updates**: Live market data integration
- **A/B Testing**: Model comparison and selection
- **Explainable AI**: SHAP values for prediction explanations

## 📚 Technical Details

### Architecture

- **Frontend**: HTML5, Bootstrap 4, jQuery
- **Backend**: Flask (Python)
- **ML Pipeline**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Deployment**: Gunicorn WSGI server

### Data Pipeline

1. **Data Loading**: CSV file processing
2. **Feature Engineering**: Automated feature creation
3. **Preprocessing**: Scaling, encoding, validation
4. **Model Training**: Multiple algorithm comparison
5. **Evaluation**: Cross-validation and metrics
6. **Deployment**: Model serialization and loading

### Security Features

- Input validation and sanitization
- Error handling without sensitive data exposure
- Request size limits
- CORS configuration ready
- Environment variable configuration

---

**Built with ❤️ for accurate car price predictions**
A web-based application for predicting car prices using machine learning algorithms. Users can input vehicle specifications, and the app provides an estimated price based on trained data. The app is built with Flask for the backend and includes user-friendly HTML templates for a seamless experience.

# Car Price Prediction

A web-based application for predicting car prices using machine learning algorithms. Users can input vehicle specifications, and the app provides an estimated price based on trained data.

## Features

- User-friendly interface for inputting car details.
- Machine learning model trained on real-world data.
- Predicts car prices based on various specifications.

## Technologies Used

- Flask (Python)
- Pandas (Data Manipulation)
- NumPy (Numerical Computation)
- Joblib (Model Serialization)

## How to Run

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`
4. Access the app at `http://127.0.0.1:5000`

## Installation

1. Clone the repository:

   git clone https://github.com/aiharshh/Car_Price_Prediction

   cd Car_Price_Prediction
