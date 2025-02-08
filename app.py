from flask import Flask, render_template
import pandas as pd
import joblib
import plotly.graph_objs as go
import plotly.offline as pyo
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the dataset and model
data = pd.read_csv('data/cleaned_gold_price_dataset.csv')
model = joblib.load('models/gold_price_model.pkl')

# Prepare the latest prediction
latest_close_price = data['close_price'].iloc[-1]
predicted_price = model.predict([[float(latest_close_price)]])[0]

# Generate the Plotly graph for price trends
def create_plot():
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['close_price'],
        mode='lines+markers',
        name='Gold Price'
    ))

    fig.update_layout(
        title='Gold Price Trend',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark'
    )

    return pyo.plot(fig, output_type='div')

@app.route('/')
def home():
    plot_div = create_plot()
    today = datetime.now().strftime('%Y-%m-%d')
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    return render_template('index.html', plot_div=plot_div, today=today, predicted_price=predicted_price, tomorrow=tomorrow)

if __name__ == '__main__':
    app.run(debug=True)