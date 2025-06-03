from flask import Blueprint, jsonify, current_app
from flask_restful import Api, Resource
import pandas as pd
import os
from pathlib import Path
# Create blueprint and API instance
comparison_api = Blueprint('comparison_api', __name__, url_prefix='/api')
api = Api(comparison_api)

class ComparisonData(Resource):
    def get(self):
        try:
            # Path to the cleaned CSV
            csv_path = os.path.join(current_app.root_path, 'data', 'comparisondata.csv')

            # Check if file exists
            if not os.path.exists(csv_path):
                return {
                    "success": False,
                    "error": "comparisondata.csv not found",
                    "path": csv_path
                }, 404

            # Load the data
            df = pd.read_csv(csv_path)
            df = df.fillna("")

            # Normalize column names
            df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_')

            return {
                "success": True,
                "count": len(df),
                "data": df.to_dict(orient="records")
            }

        except Exception as e:
            current_app.logger.exception("Failed to load comparison data")
            return {
                "success": False,
                "error": str(e),
                "type": type(e).__name__
            }, 500

# Register the resource
api.add_resource(ComparisonData, '/comparison')
