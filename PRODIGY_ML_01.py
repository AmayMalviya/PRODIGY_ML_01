import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Data Collection and Preparation.Load the dataset
data = pd.read_csv('train.csv')

# Display the first few rows of the dataset
print(data.head())

# Step 2: Data Preprocessing.Checking for missing values
print(data.isnull().sum())

# Fill missing values or drop rows/columns with missing values
data = data.dropna()

# Verify that there are no more missing values
print(data.isnull().sum())

# Step 3: Feature Selection.Features
X = data[['square_footage', 'num_bedrooms', 'num_bathrooms']]

# Target variable
y = data['price']

# Step 4: Model Training. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Step 5: Model Evaluation. Predict on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Step 6: Prediction. New data for prediction
new_data = pd.DataFrame({
    'square_footage': [2000, 3000],
    'num_bedrooms': [3, 4],
    'num_bathrooms': [2, 3]
})

# Predict the prices
predicted_prices = model.predict(new_data)

print(predicted_prices)

# Optional: Visualization of the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
