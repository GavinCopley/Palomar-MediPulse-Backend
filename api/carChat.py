import jwt
from flask import Blueprint, request, jsonify, current_app, Response, g
from flask_restful import Api, Resource  # used for REST API building
from flask import Blueprint
from __init__ import app
from flask_restful import Resource, reqparse, Api
from model.carChat import carChat
#from app import db
from api.jwt_authorize import token_required
import base64
import json


# Create a single Blueprint
carChat_api = Blueprint('carChat_api', __name__, url_prefix='/api')
api = Api(carChat_api)

class carChatResource(Resource):  # Renamed to avoid confusion with model
    def get(self):
        messages = carChat.query.all()
        return [msg.read() for msg in messages], 200

    @token_required()
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('message', required=True, help="Message cannot be blank")
        # parser.add_argument('user_id', type=int, required=True, help="User ID cannot be blank")
        
        
        args = parser.parse_args()

        current_user = g.current_user

        new_message = carChat(
            message=args['message'],
            user_id=current_user.id
        )
        try:
            new_message.create()
            return new_message.read(), 201
        except Exception as e:
            return {'error': str(e)}, 400
    
        
    @token_required()  # Ensure the user is authenticated
    def delete(self):
        # Obtain the current user
        current_user = g.current_user
        # Obtain the request data
        data = request.get_json()
        
        # Find the current message from the database table(s)
        message = carChat.query.get(data['id'])
        
        if message is None:
            return jsonify({"error": "Message not found"}), 404
        
        # Check if the current user is the owner of the message
        if message._user_id != current_user.id:
            return jsonify({"message": "You are not authorized to delete this message"}), 403
        
        # Delete the message using the ORM method defined in the model
        message.delete()
        
        # Return response
        return 201

  # Ensure the user is authenticated
    @token_required()
    def put(self):
        # Obtain the current user
        current_user = g.current_user
        # Obtain the request data
        data = request.get_json()
        
        # Find the current message from the database table(s)
        message = carChat.query.get(data['id'])
        
        if message is None:
            return jsonify({"error": "Message not found"}), 404
        
        # Check if the current user is the owner of the message
        if message._user_id != current_user.id:
            return jsonify({"message": "You are not authorized to edit this message"}), 403
        
        # Update the message content
        message._message = data['message']  # Update with new message content
        message.update()  # Call the update method to save changes
        
        # Return response with updated message data
        return jsonify(message.read()), 200


api.add_resource(carChatResource, '/carChat')

