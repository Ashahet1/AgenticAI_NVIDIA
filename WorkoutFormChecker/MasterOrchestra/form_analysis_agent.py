import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_agent import BaseAgent

class FormAnalysisAgent(BaseAgent):
    """Analyzes exercise form and identifies mechanical issues"""
    
    def __init__(self):
        super().__init__(
            name="FormAnalysisAgent",
            role="analyzing exercise form and biomechanics"
        )
    
    def execute(self, parsed_data):
        """Analyze form issues based on extracted information"""
        
        exercise = parsed_data.get("exercise", "unknown")
        pain_location = parsed_data.get("pain_location", "unspecified")
        pain_timing = parsed_data.get("pain_timing", "unspecified")
        
        prompt = f"""You are an expert strength coach analyzing form issues.

Exercise: {exercise}
Pain Location: {pain_location}
Pain Timing: {pain_timing}

Analyze the likely FORM ISSUES that could cause this pain pattern.

Provide:
1. Most likely form breakdown (1-2 sentences)
2. Biomechanical explanation (1-2 sentences)
3. What to check in form (2-3 specific points)

Be specific to this exercise and pain pattern. Keep under 200 words."""
        
        response = self.call_llm(prompt, max_tokens=400)
        
        self.log_action("analyze_form", response)
        
        return {
            "agent": self.name,
            "exercise": exercise,
            "form_analysis": response,
            "confidence": "high" if exercise != "unknown" else "low",
            "status": "success"
        }
