import joblib
from flask import Flask, jsonify, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline
pipeline = joblib.load('trained_pipeline.pkl')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict(flat=True)  
        
        input_data = pd.DataFrame([data]) 
        
        input_data = input_data.astype(float)
        
        prediction = pipeline.predict(input_data)
        
        return render_template('home.html', prediction_text=f"The house price prediction is {prediction[0]:.2f}")
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
