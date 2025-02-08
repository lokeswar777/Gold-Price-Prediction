from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and dataset
try:
    model = joblib.load('models/gold_price_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    data = pd.read_csv('data/cleaned_gold_price_dataset.csv')
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get the latest closing price
        latest_close_price = float(data['close_price'].iloc[-1])

        # Predict the next day's price
        predicted_price = model.predict([[latest_close_price]])[0]

        return jsonify({
            'today_price': latest_close_price,
            'tomorrow_predicted_price': predicted_price
        })
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/historical', methods=['GET'])
def historical_data():
    try:
        historical_prices = data[['date', 'close_price']].tail(100)  # Last 100 records
        historical_list = historical_prices.to_dict(orient='records')

        return jsonify(historical_list)
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
