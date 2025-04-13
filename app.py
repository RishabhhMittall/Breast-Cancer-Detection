from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

# Load the trained model
with open("breast_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "🏥 Breast Cancer Prediction API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        # Expecting 30 features as a list
        if "features" not in input_data:
            return jsonify({"error": "Missing 'features' key in JSON"}), 400

        features = input_data["features"]

        if not isinstance(features, list) or len(features) != 30:
            return jsonify({"error": "Input must be a list of 30 numerical values."}), 400

        input_array = np.asarray(features).reshape(1, -1)
        prediction = model.predict(input_array)

        result = "Malignant" if prediction[0] == 0 else "Benign"

        return jsonify({
            "prediction": int(prediction[0]),
            "result": f"The Breast Cancer is {result}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
