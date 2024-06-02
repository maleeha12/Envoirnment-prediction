from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the models
random_forest_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            # Get the form data
            HS_analog = float(request.form['HS_analog'])
            L_lux = float(request.form['L_lux'])
            T_deg = float(request.form['T_deg'])
            CO2_analog = float(request.form['CO2_analog'])
            HR_percent = float(request.form['HR_percent'])

            # Prepare the feature array
            features = np.array([[HS_analog, L_lux, T_deg, CO2_analog, HR_percent]])

            # Scale the features
            scaled_features = scaler.transform(features)

            # Make the prediction
            prediction = random_forest_model.predict(scaled_features)

            # Interpret the prediction
            if prediction == 1:
                result = "Dry soil"
            elif prediction == 2:
                result = "Good environment"
            elif prediction == 3:
                result = "Too hot"
            elif prediction == 4:
                result = "Too cold environment"
            else:
                result = "Unknown"
        except Exception as e:
            result = f"Error: {e}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)