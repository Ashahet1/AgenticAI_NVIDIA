# form_analysis_agent.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_agent import BaseAgent

class FormAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="FormAnalysisAgent", role="analyzing exercise form")
    
    def execute(self, collected_data):
        exercise = collected_data.get("exercise", "unknown")
        pain_location = collected_data.get("pain_location", "unspecified")
        pain_timing = collected_data.get("pain_timing", "unspecified")
        
        prompt = f"""Analyze form issues for:
Exercise: {exercise}
Pain Location: {pain_location}
Pain Timing: {pain_timing}

Analyze the likely FORM ISSUES that could cause this pain pattern.

Provide:
1. Most likely form breakdown (1-2 sentences)
2. Biomechanical explanation (1-2 sentences)
3. What to check in form (2-3 specific points)

Be specific to this exercise and pain pattern. Keep under 200 words.Don't include meta comments in final plan"""
        
        try:
            response = self.call_llm(prompt, max_tokens=300)
            return {
                "agent": self.name,
                "form_analysis": response,
                "status": "success"
            }
        except Exception as e:
            return {
                "agent": self.name,
                "form_analysis": f"Form analysis unavailable: {str(e)}",
                "status": "error"
            }