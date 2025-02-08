# scripts/fetch_and_update_data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib  # For saving the model

# Load the existing dataset
df = pd.read_csv('data/cleaned_gold_price_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

# Fetch new data
last_date = df['date'].max()
next_date = last_date + timedelta(days=1)

gold_ticker = 'GC=F'
new_data = yf.download(gold_ticker, start=next_date.strftime('%Y-%m-%d'), end=datetime.today().strftime('%Y-%m-%d'), interval='1d')

if not new_data.empty:
    new_data.reset_index(inplace=True)
    new_data = new_data[['Date', 'Open', 'High', 'Low', 'Close']]
    new_data.rename(columns={
        'Date': 'date',
        'Open': 'open_price',
        'High': 'high_price',
        'Low': 'low_price',
        'Close': 'close_price'
    }, inplace=True)

    # Append and save updated dataset
    updated_df = pd.concat([df, new_data], ignore_index=True).drop_duplicates(subset=['date'])
    updated_df.to_csv('data/cleaned_gold_price_dataset.csv', index=False)
    print("Dataset updated with the latest data.")
else:
    updated_df = df
    print("No new data available.")

# Retrain the model
updated_df['next_close_price'] = updated_df['close_price'].shift(-1)
updated_df = updated_df[:-1]

X = updated_df[['close_price']]
y = updated_df['next_close_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model retrained. RMSE: {rmse}")

# Save the model
joblib.dump(model, 'models/gold_price_model.pkl')
