from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
import os
import sys

# Adjust path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.insert(0, parent_dir)

from model.hosp import HospMLModel

hosp_api_bp = Blueprint('hosp_api_bp', __name__, url_prefix='/api')
api = Api(hosp_api_bp)

# Load the trained model
model_instance = HospMLModel()
model_instance.load_model()

class HospPredictAPI(Resource):
    def post(self):
        data = request.get_json()
        if not data:
            return {"error": "No JSON data provided"}, 400

        disease = data.get("disease")      # e.g. "Acute Stroke"
        priority = data.get("priority")    # e.g. "quality"
        distance = data.get("distance")    # e.g. 25
        limit = data.get("limit", 3)       # e.g. 3 (default to 3 if not given)

        # Basic validation
        if not disease or not priority or not distance:
            return {
                "error": "Missing required fields: disease, priority, distance"
            }, 400

        try:
            limit = int(limit)
        except ValueError:
            return {"error": "Invalid limit"}, 400

        # Call the model
        try:
            results = model_instance.predict_ranked(
                disease=disease,
                priority=priority,
                max_distance=float(distance),
                limit=limit
            )
        except Exception as e:
            return {"error": str(e)}, 500

        # If no hospitals found, return an error or an empty list
        if not results:
            return {"error": "No hospitals found for these parameters."}, 200

        # Return them in a consistent JSON structure
        return {
            "recommended_hospitals": results
        }, 200

# Add the resource to the API
api.add_resource(HospPredictAPI, "/predict")
