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
            survey = Survey(**survey_data)
            survey.create()
            return {"message": "Thank you", "survey": survey.read()}, 201
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
