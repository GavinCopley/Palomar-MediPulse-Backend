from flask import Blueprint, jsonify, current_app
from flask_restful import Api, Resource
from flask_cors import cross_origin
import pandas as pd
import os
from pathlib import Path

youtube_api = Blueprint('youtube_api', __name__, url_prefix='/api/youtube')
api = Api(youtube_api)

# Change the directory path to use data/yt instead of data/youtube
YOUTUBE_DATA_DIR = os.path.join(current_app.root_path, 'data', 'yt')
print(f"Looking for YouTube data in: {os.path.abspath(YOUTUBE_DATA_DIR)}")

def ensure_youtube_dir():
    """Ensure the YouTube data directory exists"""
    try:
        os.makedirs(YOUTUBE_DATA_DIR, exist_ok=True)
        print(f"YouTube data directory ready at {os.path.abspath(YOUTUBE_DATA_DIR)}")
        return True
    except Exception as e:
        print(f"Error creating YouTube directory: {str(e)}")
        return False

class YouTubeFiles(Resource):
    @cross_origin()
    def get(self):
        try:
            print("Received request for YouTube files")
            # Ensure directory exists before listing files
            if not os.path.exists(YOUTUBE_DATA_DIR):
                print(f"Directory does not exist: {YOUTUBE_DATA_DIR}")
                if not ensure_youtube_dir():
                    return {"success": False, "error": "Could not create YouTube data directory"}, 500
            
            # List only CSV files
            files = [f for f in os.listdir(YOUTUBE_DATA_DIR) if f.endswith('.csv')]
            print(f"Found files: {files}")
            
            # Return empty list if no files found
            if not files:
                print("No CSV files found in directory")
                return {"success": True, "files": []}
                
            return {"success": True, "files": files}
        except Exception as e:
            print(f"Error in YouTubeFiles.get(): {str(e)}")
            return {"success": False, "error": str(e)}, 500

class YouTubeData(Resource):
    @cross_origin() 
    def get(self, filename):
        try:
            print(f"Received request for YouTube data: {filename}")
            path = os.path.join(YOUTUBE_DATA_DIR, filename)
            print(f"Looking for file at: {path}")
            
            if not os.path.exists(path):
                print(f"File not found: {path}")
                return {"success": False, "error": "File not found"}, 404

            print(f"Reading CSV file: {path}")
            df = pd.read_csv(path).fillna("")
            print(f"CSV columns: {df.columns.tolist()}")
            df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_')
            print(f"Normalized columns: {df.columns.tolist()}")
            data = df.to_dict(orient="records")
            print(f"Converted {len(data)} rows to JSON")
            return {"success": True, "data": data}
        except Exception as e:
            print(f"Error in YouTubeData.get(): {str(e)}")
            return {"success": False, "error": str(e)}, 500

api.add_resource(YouTubeFiles, '/files')
api.add_resource(YouTubeData, '/data/<string:filename>')

# Add to your frontend fetch calls:
# const fetchOptions = {
#     method: 'GET',
#     mode: 'cors',
#     cache: 'default',
#     credentials: 'include',
#     headers: {
#         'Content-Type': 'application/json',
#         'X-Origin': 'client'
#     }
# };

# Example fetch call:
# const response = await fetch(`${pythonURI}/api/youtube/files`, fetchOptions);
