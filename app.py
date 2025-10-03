from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model & scaler
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")    # save scaler separately when training

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features from form
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)

        # Scale input
        features = scaler.transform(features)

        # Predict using model
        prediction = model.predict(features)[0]

        result = "Fraudulent Transaction 🚨" if prediction == 1 else "Legit Transaction ✅"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
   