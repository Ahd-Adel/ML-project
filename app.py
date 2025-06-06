from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and preprocessors
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input from form
        age = int(request.form['age'])
        billing = float(request.form['billing'])
        condition = request.form['condition']
        admission = request.form['admission']
        test_result = request.form['test_result']

        # Encode categorical inputs
        condition_enc = encoder['Medical Condition'].transform([condition])[0]
        admission_enc = encoder['Admission Type'].transform([admission])[0]
        test_result_enc = encoder['Test Results'].transform([test_result])[0]

        # Combine and scale features
        features = np.array([[age, billing, condition_enc, admission_enc, test_result_enc]])
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        return render_template('index.html', prediction_text=f"Predicted Stay Category: {prediction}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

