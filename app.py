from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open('heart_disease_svm_model.sav', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('scaler.sav', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Home route - Landing page
@app.route('/')
def home():
    return render_template('home.html')

# Test route - Prediction form
@app.route('/test')
def test():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input feature names - MUST match HTML input names
        feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                         'thalach','exang','oldpeak','slope','ca','thal']

        data = []
        for feature in feature_names:
            val = request.form.get(feature)
            if val is None or val.strip() == "":
                return f"Missing input: {feature}"
            try:
                data.append(float(val))
            except ValueError:
                return f"Invalid input for {feature}: {val}"

        # Reshape and scale input
        input_data = np.array(data).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = svm_model.predict(input_data_scaled)
        print("Prediction result:", prediction)  # Debug log

        # Interpret prediction
        if int(prediction[0]) == 1:
            result = '⚠️ High Risk: The patient shows indications of possible heart disease. Clinical evaluation is recommended.'
        else:
            result = '✅ Low Risk: No significant indicators of heart disease detected. Continue regular monitoring.'

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error occurred during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
