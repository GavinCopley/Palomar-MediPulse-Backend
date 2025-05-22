from flask import Blueprint, request, jsonify, g
from flask_restful import Api, Resource
from api.jwt_authorize import token_required
from model.rate import HospitalRating

hospital_rating_api = Blueprint('hospital_rating_api', __name__, url_prefix='/api')
api = Api(hospital_rating_api)

class HospitalRatingPost(Resource):
    @token_required()
    def post(self):
        current_user = g.current_user
        data = request.get_json()

        # Validate required fields
        if "title" not in data or "description" not in data or "hospital" not in data or "rating" not in data:
            return jsonify({'message': 'Missing required fields'}), 400

        post = HospitalRating(
            title=data['title'],
            description=data['description'],
            uid=current_user.id,
            hospital=data['hospital'],
            rating=data['rating']
        )
        post.create()
        return jsonify(post.read()), 201

api.add_resource(HospitalRatingPost, '/hospitalPost')