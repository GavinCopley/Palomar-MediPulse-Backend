import google.generativeai as genai
import os
import json
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env file. Real-time hospital info will be disabled.")

# Create a cache directory if it doesn't exist
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "hospital_info_cache.json"

class HospitalInfoEnricher:
    def __init__(self):
        """Initialize the HospitalInfoEnricher with Gemini model and cache"""
        self.enabled = API_KEY is not None
        
        if self.enabled:
            genai.configure(api_key=API_KEY)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "temperature": 0.2,  # Low temperature for factual responses
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
        
        # Load cache
        self.cache = self._load_cache()
        
    def _load_cache(self):
        """Load cached hospital information"""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cached hospital information"""
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def get_hospital_info(self, hospital_name, basic_info=None):
        """
        Get enriched information about a hospital using Gemini
        
        Args:
            hospital_name (str): Name of the hospital
            basic_info (dict): Basic hospital info from CSV to help Gemini
            
        Returns:
            dict: Enhanced hospital information
        """
        if not self.enabled:
            return {"error": "Gemini API not configured"}
        
        # Create cache key
        cache_key = f"hospital_info:{hospital_name}"
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate prompt with hospital basic info if available
        prompt = f"Provide factual information about {hospital_name}. Include: "
        prompt += "1) Notable achievements or awards, "
        prompt += "2) Recent medical technology or facility upgrades, "
        prompt += "3) Special programs they offer, "
        prompt += "4) Community outreach initiatives. "
        prompt += "Provide information in JSON format with these fields: achievements, technology, programs, community_initiatives."
        
        if basic_info:
            prompt += "\n\nHere is some basic information about the hospital:\n"
            prompt += f"Location: {basic_info.get('location', 'Unknown')}\n"
            prompt += f"Specialties: {basic_info.get('specialties', 'Unknown')}\n"
            prompt += f"Departments: {basic_info.get('departments', 'Unknown')}\n"
        
        # Retry mechanism
        max_retries = 3
        delay = 1
        response = None
        
        for _ in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                break
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                time.sleep(delay)
                delay *= 2
        
        if not response:
            return {"error": "Failed to get information from Gemini API"}
        
        # Extract JSON from response
        try:
            result_text = response.text
            # Find JSON content between triple backticks if present
            if "```json" in result_text and "```" in result_text.split("```json")[1]:
                json_content = result_text.split("```json")[1].split("```")[0].strip()
            elif "{" in result_text and "}" in result_text:
                # Extract content between first { and last }
                start = result_text.find('{')
                end = result_text.rfind('}') + 1
                json_content = result_text[start:end]
            else:
                json_content = result_text
                
            enriched_info = json.loads(json_content)
            
            # Cache the result
            self.cache[cache_key] = enriched_info
            self._save_cache()
            
            return enriched_info
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            return {"error": f"Could not parse information: {str(e)}"}