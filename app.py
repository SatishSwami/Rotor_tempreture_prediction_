from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)


sc_x = joblib.load('utils/transform.pkl')         
sc_y = joblib.load('utils/op_transform.pkl')     
model = joblib.load('utils/model.pkl')            

@app.route('/')
def home():
    return render_template('index.html')    

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')

        # Validate input
        if not features or len(features) != 6:
            return jsonify({'error': 'Invalid input. Expected 6 numeric values.'}), 400

        # Reshape and transform
        sample = np.array(features).reshape(1, -1)
        sample_scaled = sc_x.transform(sample)

        # Predict and inverse transform
        pred_scaled = model.predict(sample_scaled)
        pred_original = sc_y.inverse_transform(pred_scaled.reshape(-1, 1))

        predicted_value = round(float(pred_original[0][0]), 2)

        return jsonify({'predicted_pm': predicted_value})

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
