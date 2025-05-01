from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Create a Blueprint for the chatbot functionality
chatbot_api = Blueprint('chatbot_api', __name__, url_prefix='/api')
api = Api(chatbot_api)

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
        "max_output_tokens": 512,  # Reduced to prevent excessive responses
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=(
            "You are professionally trained in the medical field. You know the answer to every question about hospitals, diseases, and treatments."
            "You respond in concise, minimal sentences (no more than 4 sentences)."
            "You are being hosted on a website called 'MediPulse' This website is a platform for patients to find the best hospital for them based on their needs."
        ),
    )

class Chatbot(Resource):
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
api.add_resource(Chatbot, '/chatbot')
