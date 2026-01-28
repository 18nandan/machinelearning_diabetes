from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('logistic_regression_diabetes_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    # Exit or handle error appropriately in a real application
    model = None
    scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    data = request.get_json(force=True)

    # Ensure the input data has the correct feature names and order
    # Based on the original DataFrame columns, excluding 'Outcome'
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    try:
        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data])
        
        # Reorder columns to match the training data's feature order
        input_df = input_df[feature_names]

        # Scale the input data using the pre-trained scaler
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Return the prediction as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'probability_no_diabetes': float(prediction_proba[0][0]),
            'probability_diabetes': float(prediction_proba[0][1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# To run the Flask app directly from Colab, you can use ngrok or a similar tool.
# For local testing, you would typically run `app.run(debug=True, port=5000)`
# However, running Flask apps directly in Colab without specific tunneling setup
# might require additional steps.
print("Flask API '/predict' endpoint ready. You can run this cell and then use tools like ngrok to expose it.")
# To make it runnable in Colab and expose it, you would typically use ngrok
# Example (uncomment and install ngrok if you want to run this in Colab):
# from flask_ngrok import run_with_ngrok
# run_with_ngrok(app) # Starts ngrok when app is run
# app.run()
print("Flask API '/predict' endpoint ready. You can run this cell and then use tools like ngrok to expose it.")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
