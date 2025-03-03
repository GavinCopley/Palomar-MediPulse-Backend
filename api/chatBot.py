from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Create a Blueprint for the chatbot functionality
chatbot_api = Blueprint('chatbot_api', __name__, url_prefix='/api')
api = Api(chatbot_api)

# Configure the API key
genai.configure(api_key=os.getenv('CHATBOT_API_KEY'))

# Model configuration
generation_config = {
    "temperature": 1.15,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 512,  # Reduced to prevent excessive responses
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=(
        "You are a car expert and enthusiast. You know the answer to every question about cars."
        "You respond in concise, minimal sentences (no more than 4 sentences)."
    ),
)

class Chatbot(Resource):
    def post(self):
        data = request.get_json()
        user_input = data.get("user_input")

        if not user_input:
            return jsonify({"error": "User input is required"}), 400

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
api.add_resource(Chatbot, '/chatbot')
