import pickle
from flask import Flask, jsonify, request, render_template
import numpy as np

app = Flask(__name__)

model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    features = [float(request.form[x]) for x in request.form]

    # Transform the input data using the scaler
    final_input = scaler.transform(np.array(features).reshape(1, -1))

    # Make prediction using the model
    prediction = model.predict(final_input)[0]

    # Render the home.html template with prediction result
    return render_template('home.html', prediction_text=f"The house price prediction is {prediction}")

if __name__ == '__main__':
    app.run(debug=True)
