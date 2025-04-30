from flask import Blueprint, request, jsonify, g
from flask_restful import Api, Resource
from model.survey import Survey
from __init__ import db
from api.jwt_authorize import token_required  # Ensure the user is authenticated
from flask_cors import cross_origin

survey_api = Blueprint('survey_api', __name__, url_prefix='/api')
api = Api(survey_api)

from api.jwt_authorize import token_required  # Your existing custom JWT auth wrapper

from flask import request, g
from api.jwt_authorize import token_required
from model.survey import Survey

class SurveyResource(Resource):
    @token_required()
    def post(self):
        try:
            uid = g.current_user._uid  # ✅ Securely extracted from verified JWT
            print(">>> Authenticated UID:", uid)

            survey_data = request.get_json()
            print("Received Survey Data:", survey_data)

            # Validate required fields
            required_fields = ['name', 'username', 'email', 'age', 'weight', 'height', 'ethnicity']
            for field in required_fields:
                if field not in survey_data or survey_data[field] is None:
                    return {"error": f"Missing required field: {field}"}, 400

            # Check if survey already exists and is marked as completed
            existing_survey = Survey.query.filter_by(uid=uid).first()
            if existing_survey and existing_survey.survey_completed:
                return {"error": "Survey already completed by this user"}, 400

            # Create or update survey
            if existing_survey:
                # Update existing survey if not marked complete
                updated_survey = existing_survey.update({
                    "name": survey_data['name'],
                    "username": survey_data['username'],
                    "email": survey_data['email'],
                    "number": survey_data.get('number'),
                    "age": survey_data['age'],
                    "weight": survey_data['weight'],
                    "height": survey_data['height'],
                    "allergies": survey_data.get('allergies'),
                    "conditions": survey_data.get('conditions'),
                    "ethnicity": survey_data['ethnicity'],
                    "survey_completed": survey_data.get('survey_completed', False)
                })
                return {
                    "message": "Survey updated successfully",
                    "survey": updated_survey.read()
                }, 200
            else:
                # Create new survey
                new_survey = Survey(
                    uid=uid,
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
                    survey_completed=survey_data.get('survey_completed', False)
                )

                result = new_survey.create()
                if result:
                    return {
                        "message": "Survey submitted successfully!",
                        "survey": result.read()
                    }, 201
                else:
                    return {"error": "Failed to create survey (possibly duplicate username/email)."}, 400

        except Exception as e:
            print("Survey POST Error:", str(e))
            return {"error": "Internal server error", "details": str(e)}, 500

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
