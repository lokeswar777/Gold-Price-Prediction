from flask import Flask, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://lokeswar777.github.io"}})  # Allow only your GitHub Pages domain

# Load model and data
model = joblib.load('models/gold_price_model.pkl')
data = pd.read_csv('data/cleaned_gold_price_dataset.csv')

latest_close_price = data['close_price'].iloc[-1]
predicted_price = model.predict([[float(latest_close_price)]])[0]

@app.route('/predict', methods=['GET'])
def get_prediction():
    return jsonify({
        'predicted_price': predicted_price,
        
        'currency': 'INR'
    })

if __name__ == '__main__':
    app.run(debug=True)
