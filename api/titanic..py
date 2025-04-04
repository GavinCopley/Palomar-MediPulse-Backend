"""
File: titanic_api.py

Purpose:
- Provides a Flask Blueprint to handle API routes for Titanic predictions.
"""
from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource

# Import the TitanicModel from your model file
from models.titanic import TitanicModel  # adapt the path to match your project

titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)

class _Predict(Resource):
    def post(self):
        """
        POST request to predict Titanic survival.
        Expects JSON in the request body with passenger data:
        {
          "name": ["John Doe"],
          "pclass": 1,
          "sex": "male",
          "age": 30,
          "sibsp": 0,
          "parch": 0,
          "fare": 100,
          "embarked": "S",
          "alone": false
        }
        """
        passenger = request.get_json()
        titanic_model = TitanicModel.get_instance()
        result = titanic_model.predict(passenger)
        return jsonify(result)

api.add_resource(_Predict, '/predict')
