# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump  # Import dump for saving the model

# Load your cleaned data
car_data = pd.read_csv(r'C:\Users\User\Downloads\used_cars_filled.csv')

# Display the first few rows of the dataset to understand its structure
print("Dataset preview:")
print(car_data.head())

# Data exploration: Descriptive statistics
print("\nDescriptive statistics:")
print(car_data.describe())

# Visualize the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(car_data.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Prepare features and target variable
X = car_data.drop('price(in lakhs)', axis=1)  # Features
y = car_data['price(in lakhs)']  # Target variable

# Check for and handle any missing values (if necessary)
print("\nMissing values in features before splitting:")
print(X.isnull().sum())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model's performance
print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))

# Visualize actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Line for perfect prediction
plt.xlabel('Actual Prices (in lakhs)')
plt.ylabel('Predicted Prices (in lakhs)')
plt.title('Actual vs Predicted Prices')
plt.grid()
plt.show()

# Save the trained model using joblib
dump(model, 'car_price_model.joblib')  # This saves the model to a file
print("Model saved as car_price_model.joblib")
