import os
import pickle
from flask import Flask, request, render_template, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
scale = pickle.load(open(os.path.join(BASE_DIR, 'scale.pkl'), 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get input data
        input_features = [float(x) for x in request.form.values()]
        features_array = np.array([input_features])

        # Assign column names
        columns = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day',
                   'hours', 'minutes', 'seconds']
        data = pd.DataFrame(features_array, columns=columns)

        # Scale the data
        scaled_data = scale.transform(data)

        # Predict
        prediction = model.predict(scaled_data)
        predicted_value = round(prediction[0], 2)

        # Render result page
        return render_template("result.html", result=f"Estimated Traffic Volume: {predicted_value}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error occurred: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port, use_reloader=False)
