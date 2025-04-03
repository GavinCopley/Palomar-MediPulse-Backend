import jwt
from flask import Blueprint, request, jsonify, current_app, Response, g
from flask_restful import Api, Resource  # used for REST API building
from datetime import datetime
from __init__ import app
from model.post import Post
from model.channel import Channel

"""
This Blueprint object is used to define APIs for the Post model.
- Blueprint is used to modularize application files.
- This Blueprint is registered to the Flask app in main.py.
"""
dataPost_api = Blueprint('dataPost_api', __name__, url_prefix='/api')

"""
The Api object is connected to the Blueprint object to define the API endpoints.
- The API object is used to add resources to the API.
- The objects added are mapped to code that contains the actions for the API.
- For more information, refer to the API docs: https://flask-restful.readthedocs.io/en/latest/api.html
"""
api = Api(dataPost_api)

class DataPostAPI:
    """
    Define the API CRUD endpoints for the Post model.
    There are four operations that correspond to common HTTP methods:
    - post: create a new post
    - get: read posts
    - put: update a post
    - delete: delete a post
    """
    class _CRUD(Resource):
        def post(self):
            """
            Create a new post with metadata like like count, view count, video length, comment count, and date.
            """
            data = request.get_json()

            if not data:
                return {'message': 'No input data provided'}, 400
            if 'title' not in data:
                return {'message': 'Post title is required'}, 400
            if 'comment' not in data:
                return {'message': 'Post comment is required'}, 400
            if 'channel_id' not in data:
                return {'message': 'Channel ID is required'}, 400
            if 'content' not in data:
                data['content'] = {}
            if 'like_count' not in data:
                data['like_count'] = 0
            if 'view_count' not in data:
                data['view_count'] = 0
            if 'video_length' not in data:
                data['video_length'] = 0
            if 'comment_count' not in data:
                data['comment_count'] = 0
            if 'date' not in data:
                data['date'] = datetime.utcnow().isoformat()

            post = Post(
                title=data['title'],
                comment=data['comment'],
                user_id=None,  # Removed user authentication
                channel_id=data['channel_id'],
                content=data['content'],
                like_count=data['like_count'],
                view_count=data['view_count'],
                video_length=data['video_length'],
                comment_count=data['comment_count'],
                date=data['date']
            )
            post.create()
            return jsonify(post.read())

        def get(self):
            """
            Retrieve a single post by ID.
            """
            data = request.get_json()
            if data is None:
                return {'message': 'Post data not found'}, 400
            if 'id' not in data:
                return {'message': 'Post ID not found'}, 400
            
            post = Post.query.get(data['id'])
            if post is None:
                return {'message': 'Post not found'}, 404
            
            return jsonify(post.read())

        def put(self):
            """
            Update a post.
            """
            data = request.get_json()
            post = Post.query.get(data['id'])
            if post is None:
                return {'message': 'Post not found'}, 404
            
            post._title = data['title']
            post._content = data['content']
            post._channel_id = data['channel_id']
            post._like_count = data.get('like_count', post._like_count)
            post._view_count = data.get('view_count', post._view_count)
            post._video_length = data.get('video_length', post._video_length)
            post._comment_count = data.get('comment_count', post._comment_count)
            post._date = data.get('date', post._date)
            post.update()
            return jsonify(post.read())

        def delete(self):
            """
            Delete a post.
            """
            data = request.get_json()
            post = Post.query.get(data['id'])
            if post is None:
                return {'message': 'Post not found'}, 404
            
            post.delete()
            return jsonify({"message": "Post deleted"})

    class _USER(Resource):
        def get(self):
            posts = Post.query.all()
            return jsonify([post.read() for post in posts])
    
    class _BULK_CRUD(Resource):
        def post(self):
            posts = request.get_json()
            if not isinstance(posts, list):
                return {'message': 'Expected a list of post data'}, 400
            
            results = {'errors': [], 'success_count': 0, 'error_count': 0}
            with current_app.test_client() as client:
                for post in posts:
                    response = client.post('/api/dataPost', json=post)
                    if response.status_code == 200:
                        results['success_count'] += 1
                    else:
                        results['errors'].append(response.get_json())
                        results['error_count'] += 1
            return jsonify(results)
    
    class _FILTER(Resource):
        def post(self):
            data = request.get_json()
            if data is None or 'channel_id' not in data:
                return {'message': 'Channel ID not found'}, 400
            posts = Post.query.filter_by(_channel_id=data['channel_id']).all()
            return jsonify([post.read() for post in posts])
    
    api.add_resource(_CRUD, '/dataPost')
    api.add_resource(_USER, '/dataPost/user')
    api.add_resource(_BULK_CRUD, '/dataPosts')
    api.add_resource(_FILTER, '/dataPosts/filter')
