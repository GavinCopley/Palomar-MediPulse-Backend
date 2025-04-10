from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.survey import Survey
from __init__ import db

survey_api = Blueprint('survey_api', __name__, url_prefix='/api')
api = Api(survey_api)

class SurveyResource(Resource):
    def post(self):
        try:
            survey_data = request.get_json()

            # Validate required fields
            required_fields = ['name', 'username', 'email', 'age', 'weight', 'height', 'ethnicity']
            for field in required_fields:
                if field not in survey_data:
                    return {"error": f"Missing required field: {field}"}, 400

            # Create a new survey entry
            survey = Survey(
                name=survey_data['name'],
                username=survey_data['username'],
                email=survey_data['email'],
                number=survey_data.get('number'),
                age=survey_data['age'],
                weight=survey_data['weight'],
                height=survey_data['height'],
                allergies=survey_data.get('allergies'),
                conditions=survey_data.get('conditions'),
                ethnicity=survey_data['ethnicity'],
                survey_completed=survey_data.get('survey_completed', False)  # Default to False if not provided
            )

            survey.create()
            return {"message": "Survey submitted successfully!", "survey": survey.read()}, 201
        except Exception as e:
            return {"error": str(e)}, 400

    def get(self):
        surveys = Survey.query.all()
        return {"surveys": [survey.read() for survey in surveys]}, 200

    def put(self, survey_id):
        survey = Survey.query.get(survey_id)
        if not survey:
            return {"error": "Survey not found"}, 404
        
        try:
            survey_data = request.get_json()
            updated_survey = survey.update(survey_data)
            if updated_survey:
                return {"message": "Survey updated successfully", "survey": updated_survey.read()}, 200
            return {"error": "Error updating survey"}, 400
        except Exception as e:
            return {"error": str(e)}, 400

    def delete(self, survey_id):
        survey = Survey.query.get(survey_id)
        if not survey:
            return {"error": "Survey not found"}, 404
        
        try:
            survey.delete()
            return {"message": "Survey deleted successfully"}, 200
        except Exception as e:
            return {"error": str(e)}, 400

# Add routes
api.add_resource(SurveyResource, '/survey', '/survey/<int:survey_id>')
