from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load("models/logistic_regression_diabetes_model.joblib")
    scaler = joblib.load("models/scaler.joblib")

    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None


@app.route("/")
def home():
    return jsonify({"status": "Diabetes Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500

    try:
        data = request.get_json(force=True)

        feature_names = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]

        input_df = pd.DataFrame([data])
        input_df = input_df[feature_names]

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        return jsonify({
            "prediction": int(prediction[0]),
            "probability_no_diabetes": float(prediction_proba[0][0]),
            "probability_diabetes": float(prediction_proba[0][1]),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
