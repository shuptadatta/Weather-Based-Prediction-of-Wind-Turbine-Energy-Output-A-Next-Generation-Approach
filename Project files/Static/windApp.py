# Step 1: Import required libraries
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import requests
from dotenv import load_dotenv

load_dotenv()

# Step 2: Load the model and initialise flask app
app = Flask(__name__)
model = joblib.load('power_prediction.sav')


# -------------------- Routes --------------------

@app.route('/')
def home():
    """Home page - landing page with navigation"""
    return render_template('intro.html')


# Step 3: Configure app for API requests
@app.route('/predict')
def predict():
    """Render the prediction page with weather API + prediction forms"""
    return render_template('predict.html')


@app.route('/windapi', methods=['POST'])
def windapi():
    """
    Takes city name from the form, hits OpenWeatherMap API
    to get current weather conditions, and sends data back to the UI.
    """
    city = request.form.get('city')
    apikey = os.getenv('OPENWEATHER_API_KEY')
    url = "http://api.openweathermap.org/data/2.5/weather?q=" + city + "&appid=" + apikey

    resp = requests.get(url)
    resp = resp.json()

    # Check if the API returned an error
    if resp.get("cod") != 200:
        error_msg = resp.get("message", "Unable to fetch weather data. Check your API key or city name.")
        return render_template('predict.html', error=error_msg)

    temp = str(round(resp["main"]["temp"] - 273.15, 2)) + " °C"
    humid = str(resp["main"]["humidity"]) + " %"
    pressure = str(resp["main"]["pressure"]) + " mmHG"
    speed = str(resp["wind"]["speed"]) + " m/s"

    return render_template('predict.html',
                           temp=temp, humid=humid,
                           pressure=pressure, speed=speed)


# Step 4: Configure the file with predictions
@app.route('/y_predict', methods=['POST'])
def y_predict():
    """
    Takes WindSpeed, MotorTorque, RotorTorque from the UI form,
    passes them to the trained model, and returns the prediction.
    """
    x_test = [[float(x) for x in request.form.values()]]

    prediction = model.predict(x_test)
    print(prediction)
    output = prediction[0]

    return render_template('predict.html',
                           prediction_text='The energy predicted is {:.2f} KWh'.format(output))


# Step 5: Run the app
if __name__ == "__main__":
    app.run(debug=True)
