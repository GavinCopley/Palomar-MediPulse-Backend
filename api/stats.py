from flask import Blueprint, jsonify
from flask_restful import Api, Resource
from api.jwt_authorize import token_required
import random

stats_api = Blueprint('stats_api', __name__)  # Remove the prefix
api = Api(stats_api)

class UserStats(Resource):
    @token_required
    def get(current_user, self):
        # Simulate data
        growth = random.randint(-10, 60)  # Growth from previous week
        post_count = random.randint(1, 7)  # Posts this week

        return jsonify({
            "name": current_user.get('name') or current_user.get('username'),
            "weeklyGrowth": growth,
            "postsThisWeek": post_count
        })

api.add_resource(UserStats, '/users/activity')  # Changed endpoint to '/users/activity'
