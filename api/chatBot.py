from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Create a Blueprint for the chatbot functionality
chatbot_api = Blueprint('chatbot_api', __name__, url_prefix='/api')
api = Api(chatbot_api)

# Configure the API key (ensure the API_KEY environment variable is set)
genai.configure(api_key=os.getenv('CHATBOT_API_KEY'))

# Create the model with the configuration
generation_config = {
    "temperature": 1.15,  # creativity of response
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,  # max size of response
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=(
        "You are a car expert and enthusiast. You know the answer to every question about cars."
        "You speak in concise and clear sentences and maintain MINIMAL sentences."
        "You DO NOT give responses longer than 4 sentences."
    ),
)

class Chatbot(Resource):
    def post(self):
        data = request.get_json()
        user_input = data.get("user_input")

        if not user_input:
            return jsonify({"error": "User input is required"}), 400

        try:
            # Start a new chat session
            chat_session = model.start_chat(history=[])
            
            # Get the response from the model
            response = chat_session.send_message(user_input)
            model_response = response.text.strip()

            return jsonify({
                "user_input": user_input,
                "model_response": model_response,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Add the resource to the API
api.add_resource(Chatbot, '/chatbot')
