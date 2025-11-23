import streamlit as st
import pandas as pd
import pickle
from prophet.serialize import model_from_json
import numpy as np
from tensorflow.keras.models import model_from_json as keras_model_from_json
import os
import altair as alt


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Load Forecasting", layout="wide")
st.title("⚡ Load Forecasting Dashboard")


# ---------------- MODEL SELECTION ----------------
st.header("Select Model View Mode")
model_options = ["Arima Model", "LSTM Model", "Prophet Model", "Hybrid Model"]
selected_model_name = st.selectbox("Choose model for detailed view:", model_options)

model_files = {
    "Arima Model": "arima.pkl",
    "LSTM Model": "lstm.pkl",
    "Prophet Model": "prophet_model.json",
    "Hybrid Model": "hybrid.pkl"
}

data_files = {
    "Arima Model": "arima.csv",
    "LSTM Model": "lstm.csv",
    "Prophet Model": "prophet.csv",
    "Hybrid Model": "hybrid.csv"
}

# Load selected model
try:
    if selected_model_name == "Prophet Model":
        with open(model_files[selected_model_name], "r") as f:
            model = model_from_json(f.read())
    else:
        with open(model_files[selected_model_name], "rb") as f:
            model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model!\n{e}")
    st.stop()

df = pd.read_csv(data_files[selected_model_name])
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Load performance data
perf_df = pd.read_csv("performance.csv") if os.path.exists("performance.csv") else None

# Load scaler if available
try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
except:
    scaler = None


# ---------------- MODE SELECTION ----------------
mode = st.radio("Choose Mode", ["Compare with Actuals", "Future Prediction"])


# ---------------- COMPARE WITH ACTUALS ----------------
if mode == "Compare with Actuals":
    days = st.slider("Select number of days", 1, 7, 1)
    rows = days * 24
    df_days = df.iloc[:rows]

    plot_df = df_days.melt(
        id_vars=['Datetime'],
        value_vars=['Actual_Load', 'Predicted_Load'],
        var_name='Type',
        value_name='Load'
    )

    plot_df['Type'] = plot_df['Type'].replace({
        'Actual_Load': 'Actual',
        'Predicted_Load': 'Predicted'
    })

    chart = alt.Chart(plot_df).mark_line().encode(
        x='Datetime:T',
        y='Load:Q',
        color='Type:N',
        tooltip=['Datetime:T', 'Load:Q', 'Type:N']
    ).properties(
        width=1000,
        height=400,
        title=f"{selected_model_name} - Actual vs Predicted"
    )

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(df_days[['Datetime', 'Actual_Load', 'Predicted_Load']])

    if perf_df is not None:
        model_key = selected_model_name.split()[0].lower()
        model_perf = perf_df[perf_df['Model'].str.lower() == model_key]
        if not model_perf.empty:
            st.subheader("Performance Metrics")
            st.write(f"MAE: {model_perf['MAE'].values[0]:.4f}")
            st.write(f"RMSE: {model_perf['RMSE'].values[0]:.4f}")
            st.write(f"MAPE: {model_perf['MAPE'].values[0]:.2f}%")
        else:
            st.warning("Performance metrics not found.")


# ---------------- FUTURE PREDICTION ----------------
elif mode == "Future Prediction":
    days = st.slider("Select number of prediction days", 1, 7, 1)
    steps = days * 24
    forecast_values = None

    if selected_model_name == "Arima Model":
        forecast = model.forecast(steps=steps)
        forecast_values = forecast.to_numpy().flatten()

    elif selected_model_name == "Prophet Model":
        future = model.make_future_dataframe(periods=steps, freq='H')
        forecast = model.predict(future)
        forecast_values = forecast['yhat'].iloc[-steps:].values

    elif selected_model_name == "LSTM Model":
        if scaler is None:
            st.error("Scaler file missing.")
            st.stop()

        window = 24
        values = df['Actual_Load'].values.reshape(-1, 1)
        scaled = scaler.transform(values)
        seq = scaled[-window:].reshape(1, window, 1)

        preds = []
        for _ in range(steps):
            pred = model.predict(seq, verbose=0)
            preds.append(pred[0, 0])
            seq = np.append(seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        forecast_values = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    elif selected_model_name == "Hybrid Model":
        hybrid_data = model
        lstm_json = hybrid_data["lstm_json"]
        lstm_weights = hybrid_data["lstm_weights"]
        lookback = hybrid_data["lookback"]
        scaler = hybrid_data["scaler"]
        prophet_forecast = hybrid_data["prophet_forecast"]

        lstm_model = keras_model_from_json(lstm_json)
        lstm_model.set_weights([np.array(w) for w in lstm_weights])

        last_vals = df['Actual_Load'].values[-lookback:]
        seq = scaler.transform(last_vals.reshape(-1, 1)).reshape(1, lookback, 1)

        future_lstm = []
        for _ in range(steps):
            pred_scaled = lstm_model.predict(seq, verbose=0)
            pred = scaler.inverse_transform(pred_scaled).flatten()[0]
            future_lstm.append(pred)
            seq = np.append(seq[:, 1:, :],
                            scaler.transform([[pred]]).reshape(1, 1, 1),
                            axis=1)

        prophet_pred = np.array(prophet_forecast["yhat"])
        n = min(len(future_lstm), len(prophet_pred), steps)
        forecast_values = (np.array(future_lstm[:n]) + prophet_pred[:n]) / 2

    future_index = pd.date_range(
        start=df['Datetime'].iloc[-1] + pd.Timedelta(hours=1),
        periods=len(forecast_values),
        freq='H'
    )

    forecast_df = pd.DataFrame({
        'Datetime': future_index,
        'Load': forecast_values,
        'Type': 'Predicted'
    })

    st.altair_chart(
        alt.Chart(forecast_df).mark_line().encode(
            x='Datetime:T',
            y='Load:Q',
            color='Type:N'
        ).properties(
            width=1000, height=400,
            title=f"{selected_model_name} Forecast - Next {days} Days"
        ),
        use_container_width=True
    )

    st.dataframe(forecast_df[['Datetime', 'Load']])


st.markdown("---")


# ---------------- ALL MODELS COMBINED ----------------
st.header("📊 Actual vs All Models")

combined_files = {
    "Arima": "arima.csv",
    "LSTM": "lstm.csv",
    "Prophet": "prophet.csv",
    "Hybrid": "hybrid.csv"
}

combined_df = pd.DataFrame()

for name, file in combined_files.items():
    if os.path.exists(file):
        temp = pd.read_csv(file)
        temp['Datetime'] = pd.to_datetime(temp['Datetime'])
        temp = temp.rename(columns={'Predicted_Load': name})

        if combined_df.empty:
            combined_df = temp[['Datetime', 'Actual_Load', name]]
        else:
            combined_df = pd.merge(
                combined_df, temp[['Datetime', name]],
                on='Datetime', how='outer'
            )

combined_df = combined_df.sort_values('Datetime')

# 🔹 New Days slider added
days_all = st.slider("Select number of days to display", 1, 7, 1, key="all_models_slider")
subset_all = combined_df.head(days_all * 24)

melt_all = subset_all.melt(
    id_vars=['Datetime'],
    value_vars=['Actual_Load', 'Arima', 'LSTM', 'Prophet', 'Hybrid'],
    var_name='Type', value_name='Load'
)

st.altair_chart(
    alt.Chart(melt_all).mark_line().encode(
        x='Datetime:T',
        y='Load:Q',
        color='Type:N'
    ).properties(
        width=1100,
        height=450,
        title=f"Actual vs All Model Predictions - Last {days_all} Days"
    ),
    use_container_width=True
)

st.subheader("Displayed Data")
st.dataframe(subset_all)
