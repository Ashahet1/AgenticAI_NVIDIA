
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_agent import BaseAgent

class InjuryDiagnosisAgent(BaseAgent):
    """Diagnoses injury patterns and identifies root causes"""
    
    def __init__(self):
        super().__init__(
            name="InjuryDiagnosisAgent",
            role="diagnosing injury patterns and determining root causes"
        )
    
    def execute(self, parsed_data, form_analysis):
        """Diagnose injury based on symptoms and form analysis"""
        
        exercise = parsed_data.get("exercise", "unknown")
        pain_location = parsed_data.get("pain_location", "unspecified")
        pain_timing = parsed_data.get("pain_timing", "unspecified")
        pain_intensity = parsed_data.get("pain_intensity", "unspecified")
        
        form_issues = form_analysis.get("form_analysis", "No form analysis available")
        
        prompt = f"""You are an expert sports medicine professional diagnosing workout injuries.

SYMPTOMS:
- Exercise: {exercise}
- Pain Location: {pain_location}
- Pain Timing: {pain_timing}
- Pain Intensity: {pain_intensity}

FORM ANALYSIS:
{form_issues}

Based on this information, provide:

1. MOST LIKELY INJURY/ISSUE (1 sentence)
2. ROOT CAUSE EXPLANATION (2-3 sentences connecting form + symptoms)
3. CONFIDENCE LEVEL (High/Medium/Low with brief reason)
4. RED FLAGS (if any serious concerns mention them)

Be specific and evidence-based. Keep under 250 words."""
        
        response = self.call_llm(prompt, max_tokens=500)
        
        confidence = "medium"
        if "high confidence" in response.lower() or "likely" in response.lower():
            confidence = "high"
        elif "low confidence" in response.lower() or "unclear" in response.lower():
            confidence = "low"
        
        self.log_action("diagnose_injury", response)
        
        return {
            "agent": self.name,
            "diagnosis": response,
            "confidence": confidence,
            "requires_medical_attention": "red flag" in response.lower() or "see a doctor" in response.lower(),
            "status": "success"
        }
