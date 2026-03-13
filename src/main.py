import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__, static_folder='statics')

# Load the trained Random Forest model and scaler
artifacts = joblib.load('my_model.pkl')
model = artifacts['model']
scaler = artifacts['scaler']

class_labels = ['Setosa', 'Versicolor', 'Virginica']


@app.route('/')
def home():
    return "Welcome to the Iris Classifier API!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form
            sepal_length = float(data['sepal_length'])
            sepal_width = float(data['sepal_width'])
            petal_length = float(data['petal_length'])
            petal_width = float(data['petal_width'])

            # Apply the same scaler used during training
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_data)

            # Perform the prediction
            predicted_index = model.predict(input_scaled)[0]
            predicted_class = class_labels[predicted_index]

            return jsonify({"predicted_class": predicted_class})
        except Exception as e:
            return jsonify({"error": str(e)})
    elif request.method == 'GET':
        return render_template('predict.html')
    else:
        return "Unsupported HTTP method"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)
