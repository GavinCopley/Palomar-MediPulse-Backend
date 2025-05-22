from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Create a Blueprint for the chatbot functionality
videoStore_api = Blueprint('videoStore_api', __name__, url_prefix='/api')
api = Api(videoStore_api)

# Check for API key first
api_key = os.getenv('CHATBOT_API_KEY')
if not api_key:
    print("WARNING: CHATBOT_API_KEY not found in .env file. Chatbot functionality will be limited.")
    # Set a flag to indicate that the chatbot is in limited functionality mode
    chatbot_enabled = False
else:
    chatbot_enabled = True
    # Configure Gemini API
    genai.configure(api_key=api_key)

    # Model configuration
    generation_config = {
        "temperature": 1.15,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=(
            "DescriptionThis video will explore [IDEA]. It's designed to provide valuable insights and practical information for viewers interested in this topic. Key PointsKey point 1 about [IDEA] Key point 2 about [IDEA] Key point 3 about [IDEA] StructureIntroduction and hookOverview of [IDEA]Main content section Summary of key takeawaysClosing thoughts and call to actionCall To ActionThank viewers for watching, ask them to like and subscribe, and invite them to comment with questions or share their experiences."
        ),
    )

class videoStore(Resource):
    def post(self):
        data = request.get_json()
        user_input = data.get("user_input")

        if not user_input:
            return jsonify({"error": "User input is required"}), 400

        if not chatbot_enabled:
            return jsonify({
                "user_input": user_input,
                "model_response": "I'm sorry, the chatbot is currently unavailable. Please try again later."
            }), 200

        try:
            # Get the response from the model
            response = model.generate_content(user_input)
            model_response = response.text.strip() if response and response.text else "No response generated."

            return jsonify({
                "user_input": user_input,
                "model_response": model_response,
            })
        except Exception as e:
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# Add the resource to the API
api.add_resource(videoStore, '/videoStoreAI')
