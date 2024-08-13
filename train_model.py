import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load your dataset
data = pd.read_csv('house_prices.csv')

# Data preprocessing (drop rows with missing values)
data = data.dropna()

# Select relevant features for the model
X = data[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']]
y = data['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'model/house_price_model.pkl')

print("Model has been trained and saved to 'model/house_price_model.pkl'")
