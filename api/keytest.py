# Add this to a test file or run in Python console
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
key = os.getenv('GEMINI_API_KEY')
print(f"API key found: {'Yes' if key else 'No'}")

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content('Hello world!')
print(response.text)