# 🩺 Diabetes Prediction API (Flask + Machine Learning)

This project is a **machine learning–powered REST API** that predicts whether a person is likely to have diabetes based on medical input features.

The API is built using **Flask** and a **Logistic Regression model**, and is designed to be deployed on cloud platforms like **Render**.

---

## 🚀 What This Project Does

- Takes health parameters as JSON input
- Uses a trained ML model to predict diabetes risk
- Returns:
  - Prediction (Diabetes / No Diabetes)
  - Probability scores for both outcomes
- Exposes everything through a simple `/predict` REST API

---

## 🧠 Machine Learning Model

- **Algorithm:** Logistic Regression  
- **Preprocessing:** StandardScaler  
- **Training Dataset:** Pima Indians Diabetes Dataset  
- **Features Used:**
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age

The trained model and scaler are saved using `joblib`.

---

## 🛠 Tech Stack

- **Python**
- **Flask** – API framework
- **scikit-learn** – ML model
- **pandas / numpy** – data handling
- **joblib** – model persistence
- **Gunicorn** – production server
- **Render** – cloud deployment

---

## 📁 Project Structure

