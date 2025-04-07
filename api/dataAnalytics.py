from flask import Blueprint, jsonify, current_app
from flask_restful import Api, Resource
import pandas as pd
import os
from pathlib import Path

analytics_api = Blueprint('analytics_api', __name__, url_prefix='/api')
api = Api(analytics_api)

class HospitalData(Resource):
    def get(self):
        try:
            # Get absolute path to CSV
            csv_path = os.path.join(current_app.root_path, 'data', 'hospitaldatamodified.csv')
            
            if not os.path.exists(csv_path):
                current_app.logger.error(f"CSV not found at: {csv_path}")
                return {
                    "success": False,
                    "error": "Data file not found",
                    "path": csv_path
                }, 404

            # Read CSV with explicit error handling
            df = pd.read_csv(csv_path)
            df = df.fillna("")
            
            # Convert all column names to consistent format
            df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_')
            
            return {
                "success": True,
                "count": len(df),
                "data": df.to_dict(orient='records')
            }
            
        except Exception as e:
            current_app.logger.exception("API Error")
            return {
                "success": False,
                "error": str(e),
                "type": type(e).__name__
            }, 500

api.add_resource(HospitalData, '/analytics')