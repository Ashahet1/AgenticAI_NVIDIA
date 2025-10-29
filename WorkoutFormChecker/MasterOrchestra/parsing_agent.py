
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_agent import BaseAgent
import json

class ParsingAgent(BaseAgent):
    """Extracts structured information from user's workout issue description"""
    
    def __init__(self):
        super().__init__(
            name="ParsingAgent",
            role="extracting and structuring workout issue information"
        )
    
    def execute(self, user_input):
        """Parse user input and extract key information"""
        
        prompt = f"""Extract key information from this workout issue description.

User Input: "{user_input}"

Extract and respond ONLY with valid JSON (no extra text):
{{
    "exercise": "name of exercise (e.g., squat, deadlift, bench press)",
    "pain_location": "specific body part (e.g., right knee, lower back)",
    "pain_timing": "when pain occurs (e.g., during ascent, at bottom, after workout)",
    "pain_intensity": "severity if mentioned (e.g., sharp, dull, mild, severe)",
    "additional_context": "any other relevant info"
}}"""
        
        response = self.call_llm(prompt, max_tokens=300)
        
        # Try to parse JSON
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                parsed_data = json.loads(json_str)
            else:
                parsed_data = {
                    "exercise": "unknown",
                    "pain_location": "unspecified",
                    "pain_timing": "unspecified"
                }
        except Exception as e:
            parsed_data = {
                "exercise": "unknown",
                "pain_location": "unspecified",
                "pain_timing": "unspecified",
                "error": str(e)
            }
        
        self.log_action("parse_user_input", parsed_data)
        
        return {
            "agent": self.name,
            "raw_response": response,
            "parsed_data": parsed_data,
            "status": "success"
        }
