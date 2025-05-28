import google.generativeai as genai
import os
import json
import re
from pathlib import Path
import time
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hospital_info')

# Load environment variables
load_dotenv()

# Configure Gemini
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    logger.warning("GEMINI_API_KEY not found in .env file. Real-time hospital info will be disabled.")

# Create a cache directory if it doesn't exist
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "hospital_info_cache.json"
CACHE_TTL = 86400  # Cache validity in seconds (24 hours)

class HospitalInfoEnricher:
    def __init__(self):
        """Initialize the HospitalInfoEnricher with Gemini model and cache"""
        self.enabled = API_KEY is not None
        
        if self.enabled:
            try:
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
                logger.info("Gemini API configured successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {e}")
                self.enabled = False
        
        # Load cache
        self.cache = self._load_cache()
        
    def _load_cache(self):
        """Load cached hospital information"""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                    # Clean expired cache entries
                    current_time = time.time()
                    clean_cache = {}
                    expired_count = 0
                    for key, value in cache_data.items():
                        if 'timestamp' not in value or current_time - value['timestamp'] < CACHE_TTL:
                            if 'timestamp' in value:
                                # Keep the existing entry
                                clean_cache[key] = value
                            else:
                                # Add timestamp to old entries
                                value['timestamp'] = current_time
                                clean_cache[key] = value
                        else:
                            expired_count += 1
                    
                    if expired_count > 0:
                        logger.info(f"Removed {expired_count} expired cache entries")
                        # Save the cleaned cache
                        with open(CACHE_FILE, 'w') as f:
                            json.dump(clean_cache, f)
                    
                    return clean_cache
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cached hospital information"""
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _format_response(self, data):
        """Format the response to ensure consistent structure"""
        # Ensure each field is an array of strings for frontend consistency
        result = {}
        
        for key in ['achievements', 'technology', 'programs', 'community_initiatives']:
            if key in data:
                value = data[key]
                # If it's a string, convert to a list with one element
                if isinstance(value, str):
                    result[key] = [value]
                # If it's already a list, keep it
                elif isinstance(value, list):
                    # Convert any objects to strings
                    result[key] = [str(item) if not isinstance(item, str) else item for item in value]
                else:
                    # Convert anything else to a string in a list
                    result[key] = [str(value)]
            else:
                # Provide default empty list if key is missing
                result[key] = []
                
        return result
    
    def get_hospital_info(self, hospital_name, basic_info=None, refresh=False):
        """
        Get enriched information about a hospital using Gemini
        
        Args:
            hospital_name (str): Name of the hospital
            basic_info (dict): Basic hospital info from CSV to help Gemini
            refresh (bool): Force refresh the cache
            
        Returns:
            dict: Enhanced hospital information
        """
        if not self.enabled:
            return {"error": "Gemini API not configured"}
        
        # Create cache key
        cache_key = f"hospital_info:{hospital_name}"
        
        # Check cache first unless refresh is requested
        current_time = time.time()
        if not refresh and cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            # Check if cache entry is still valid
            if 'timestamp' not in cache_entry or current_time - cache_entry['timestamp'] < CACHE_TTL:
                logger.info(f"Cache hit for hospital: {hospital_name}")
                return self._format_response(cache_entry)
            else:
                logger.info(f"Cache expired for hospital: {hospital_name}")
        
        logger.info(f"Fetching real-time info for hospital: {hospital_name}")
        
        # Generate prompt with hospital basic info if available
        prompt = f"Provide factual information about {hospital_name}. Include: "
        prompt += "1) Notable achievements or awards, "
        prompt += "2) Recent medical technology or facility upgrades, "
        prompt += "3) Special programs they offer, "
        prompt += "4) Community outreach initiatives. "
        prompt += "Provide information in JSON format with these fields: achievements (array of strings), "
        prompt += "technology (array of strings), programs (array of strings), community_initiatives (array of strings)."
        prompt += "Make sure all fields are arrays of strings, not objects or single strings."
        
        if basic_info:
            prompt += "\n\nHere is some basic information about the hospital:\n"  
            prompt += f"Location: {basic_info.get('location', 'Unknown')}\n"
            prompt += f"Specialties: {basic_info.get('specialties', 'Unknown')}\n"
            prompt += f"Departments: {basic_info.get('departments', 'Unknown')}\n"
        
        # Retry mechanism
        max_retries = 3
        delay = 1
        response = None
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                break
            except Exception as e:
                logger.error(f"Error calling Gemini API (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
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
            elif "```" in result_text and "{" in result_text.split("```")[1]:
                # Extract between ``` marks
                json_content = re.search(r'```(.*?)```', result_text, re.DOTALL).group(1).strip()
            elif "{" in result_text and "}" in result_text:
                # Extract content between first { and last }
                start = result_text.find('{')
                end = result_text.rfind('}') + 1
                json_content = result_text[start:end]
            else:
                json_content = result_text
            
            # Clean the JSON content
            json_content = json_content.replace('\n', ' ').strip()
            
            enriched_info = json.loads(json_content)
            
            # Add timestamp for cache TTL
            enriched_info['timestamp'] = current_time
            
            # Format for frontend consistency
            result = self._format_response(enriched_info)
            
            # Cache the result with timestamp
            self.cache[cache_key] = enriched_info
            self._save_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            logger.debug(f"Raw response: {response.text if response else 'None'}")
            return {"error": f"Could not parse information: {str(e)}"}
    
    def clear_cache(self, hospital_name=None):
        """
        Clear cache for a specific hospital or all hospitals
        
        Args:
            hospital_name (str, optional): Hospital name to clear cache for. If None, clears all cache.
        """
        if hospital_name:
            cache_key = f"hospital_info:{hospital_name}"
            if cache_key in self.cache:
                del self.cache[cache_key]
                logger.info(f"Cleared cache for hospital: {hospital_name}")
        else:
            self.cache = {}
            logger.info("Cleared all hospital cache")
        
        self._save_cache()
        return {"status": "success", "message": f"Cache cleared for {'all hospitals' if not hospital_name else hospital_name}"}