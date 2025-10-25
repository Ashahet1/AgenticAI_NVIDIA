
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
        
        prompt = f"""
You are a certified sports medicine professional and biomechanical analyst specializing in diagnosing exercise-related injuries.

INPUT DATA:
- Exercise performed: {exercise or 'Unknown'}
- Pain location: {pain_location or 'Not specified'}
- Pain timing (when during the motion it occurs): {pain_timing or 'Not specified'}
- Reported pain intensity (if available): {pain_intensity or 'Unknown'}

FORM & MOVEMENT ANALYSIS:
{form_issues or 'No form data provided'}

---

### TASK
Using the above data, produce a precise, evidence-based injury hypothesis. Your reasoning should reflect knowledge of biomechanics, musculoskeletal anatomy, and training form analysis.

Provide the following structured output:

1. **Probable Diagnosis:** Name the most likely injury or tissue involved (muscle, tendon, ligament, or joint) and specify the mechanism (e.g., impingement, strain, tendinitis).
2. **Root Cause (Biomechanical Explanation):** Explain *why* this injury likely occurred in relation to form errors, muscle imbalances, or overuse. Link specific movement patterns to stress points.
3. **Confidence Level:** High / Medium / Low — include a brief rationale for the rating (e.g., data clarity, pattern match strength).
4. **Red Flags:** Indicate any concerning symptoms that would require stopping exercise or seeking clinical evaluation.
5. **Clinical Insight:** Suggest what an in-person assessment would focus on (e.g., range of motion test, muscle strength imbalance, palpation zones).

Keep it under 220 words, using clear and professional tone (as if writing a quick note in a sports injury evaluation record).
"""

        
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
