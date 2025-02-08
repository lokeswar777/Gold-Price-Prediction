from flask import Flask, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and data
model = joblib.load('models/gold_price_model.pkl')
data = pd.read_csv('data/cleaned_gold_price_dataset.csv')
latest_close_price = data['close_price'].iloc[-1]

# Predict the next day's price
predicted_price = model.predict([[float(latest_close_price)]])[0]

@app.route('/predict', methods=['GET'])
def get_prediction():
    return jsonify({
        'predicted_price': predicted_price,
        'currency': 'INR'
    })

if __name__ == '__main__':
    app.run(debug=True)
