from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)


# Load the model and dataset
model = joblib.load('models/gold_price_model.pkl')
data = pd.read_csv('data/cleaned_gold_price_dataset.csv')

@app.route('/predict', methods=['GET'])
def predict():
    latest_close_price = data['close_price'].iloc[-1]
    predicted_price = model.predict([[latest_close_price]])[0]

    return jsonify({
        'today_price': float(latest_close_price),
        'tomorrow_predicted_price': float(predicted_price)
    })
@app.route('/historical', methods=['GET'])
def historical():
    historical_data = data[['date', 'close_price']].to_dict(orient='records')
    return jsonify(historical_data)


if __name__ == '__main__':
    app.run(debug=True)
