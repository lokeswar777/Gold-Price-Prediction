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
        'today_price': latest_close_price,
        'tomorrow_predicted_price': predicted_price,
        'currency': 'INR'
    })

@app.route('/historical', methods=['GET'])
def get_historical_data():
    historical_data = data[['date', 'close_price']].tail(100)  # Last 100 days
    historical_data['date'] = pd.to_datetime(historical_data['date']).dt.strftime('%Y-%m-%d')
    return historical_data.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
