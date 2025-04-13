from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

# Load the trained model
with open("breast_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)

# Define expected feature names
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

@app.route("/")
def home():
    return "🏥 Breast Cancer Prediction API is live with named features!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        if not all(feature in input_data for feature in FEATURE_NAMES):
            missing = [f for f in FEATURE_NAMES if f not in input_data]
            return jsonify({"error": f"Missing input features: {missing}"}), 400

        # Create input array using the correct order
        input_values = [float(input_data[feature]) for feature in FEATURE_NAMES]
        input_array = np.array(input_values).reshape(1, -1)

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
