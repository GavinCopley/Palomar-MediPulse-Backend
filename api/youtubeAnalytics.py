from flask import Blueprint, jsonify, request, current_app
from flask_restful import Api, Resource
import pandas as pd
import os

youtube_api = Blueprint('youtube_api', __name__, url_prefix='/api')
api = Api(youtube_api)

# Allowed CSVs (without extension)
ALLOWED_CHANNELS = {
    "Chubbyemudata",
    "Cleveland_Clinic",
    "DoctorMikedata",
    "HoustonMethodistdata",
    "Mass_General_Brigham",
    "Mayoclinic",
    "MonetefioreHealthdata",
    "NutritionFactsdata",
    "RushMedicalCenterdata",
    "ScrippsHealth",
    "Seattle",
    "TheHolisticPsychologistdata",
    "UCLAhealthdata",
    "UCSD",
    "UCSDhealthdata",
    "WhatIveLearneddata"
}

class YouTubeData(Resource):
    def get(self):
        channel = request.args.get("channel")

        if not channel or channel not in ALLOWED_CHANNELS:
            return {
                "success": False,
                "error": "Invalid or missing channel parameter.",
                "valid_channels": list(ALLOWED_CHANNELS)
            }, 400

        try:
            data_dir = os.path.join(current_app.root_path, 'data')
            csv_path = os.path.join(data_dir, f"{channel}.csv")

            if not os.path.exists(csv_path):
                return {
                    "success": False,
                    "error": f"CSV not found for channel: {channel}"
                }, 404

            df = pd.read_csv(csv_path)
            df = df.fillna("")
            df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_')

            return {
                "success": True,
                "channel": channel,
                "count": len(df),
                "data": df.to_dict(orient='records')
            }

        except Exception as e:
            current_app.logger.exception("Error reading YouTube data")
            return {
                "success": False,
                "error": str(e),
                "type": type(e).__name__
            }, 500

# Register the resource
api.add_resource(YouTubeData, '/youtube')
