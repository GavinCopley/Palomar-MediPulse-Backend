from flask import Blueprint, request, jsonify, g
from flask_restful import Api, Resource
from model.survey import Survey
from __init__ import db
from api.jwt_authorize import token_required  # Ensure the user is authenticated
from flask_cors import cross_origin

survey_api = Blueprint('survey_api', __name__, url_prefix='/api')
api = Api(survey_api)

class SurveyResource(Resource):
    def post(self):
        try:
            # Get JSON data from the request
            survey_data = request.get_json()

            # Debug: Print the incoming survey data for verification
            print("Received Survey Data:", survey_data)

            # Ensure the UID is provided in the request data
            if 'uid' not in survey_data:
                return {"error": "Missing required field: uid"}, 400

            uid = survey_data['uid']  # Get UID from the request payload

            # Validate required fields
            required_fields = ['name', 'email', 'age', 'weight', 'height', 'ethnicity']
            for field in required_fields:
                if field not in survey_data:
                    return {"error": f"Missing required field: {field}"}, 400

            # Check if the user has already completed the survey
            existing_survey = Survey.query.filter_by(uid=uid).first()
            if existing_survey and existing_survey.survey_completed:
                return {"error": "Survey already completed by this user"}, 400

            # Explicitly set survey_completed to False for new surveys if not provided
            survey_completed = survey_data.get('survey_completed', False)  # Default to False if not passed

            # Create a new survey entry
            survey = Survey(
                uid=uid,  # Set UID from the request data
                name=survey_data['name'],
                username=survey_data['username'],  # You may want to use 'username' from the request or other source
                email=survey_data['email'],
                number=survey_data.get('number'),
                age=survey_data['age'],
                weight=survey_data['weight'],
                height=survey_data['height'],
                allergies=survey_data.get('allergies'),
                conditions=survey_data.get('conditions'),
                ethnicity=survey_data['ethnicity'],
                survey_completed=survey_completed  # Set to False if it's a new survey
            )

            # Attempt to create the survey entry in the database
            survey.create()

            if survey:
                return {"message": "Survey submitted successfully!", "survey": survey.read()}, 201
            return {"error": "Failed to create survey"}, 400

        except Exception as e:
            # Log the error for debugging
            print("Error:", str(e))
            return {"error": str(e)}, 400

    def get(self, username=None):
        if username:
            # Fetch survey by username
            survey = Survey.query.filter_by(username=username).first()
            if not survey:
                return {"error": "Survey not found"}, 404

            return {
                "survey": {
                    "age": survey.age,
                    "height": survey.height,
                    "weight": survey.weight,
                    "ethnicity": survey.ethnicity,
                    "allergies": survey.allergies,
                    "conditions": survey.conditions
                }
            }, 200
        else:
            # Fetch all surveys if no username is provided
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
api.add_resource(SurveyResource, '/survey', '/survey/<int:survey_id>', '/survey/username/<string:username>')
