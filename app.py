from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://lokeswar777.github.io"}})

# Load model and dataset
model = joblib.load('models/gold_price_model.pkl')
data = pd.read_csv('data/cleaned_gold_price_dataset.csv')

# Today's price (latest close price)
latest_close_price = data['close_price'].iloc[-1]

# Predict tomorrow's price
predicted_price = model.predict([[float(latest_close_price)]])[0]

@app.route('/predict', methods=['GET'])
def get_prediction():
    return jsonify({
       
