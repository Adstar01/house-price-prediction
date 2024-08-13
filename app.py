from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model/house_price_model.pkl')

@app.route('/')
def home():
    return "Welcome to the House Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data from POST request
    
    # Extract features from the data
    features = np.array([
        data['OverallQual'],
        data['GrLivArea'],
        data['GarageCars'],
        data['TotalBsmtSF']
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return the result as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
