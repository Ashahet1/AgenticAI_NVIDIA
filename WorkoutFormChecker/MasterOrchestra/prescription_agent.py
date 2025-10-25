
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_agent import BaseAgent

class PrescriptionAgent(BaseAgent):
    
    def __init__(self):
        super().__init__(
            name="PrescriptionAgent",
            role="creating personalized action plans"
        )
    
    def execute(self, diagnosis, research, form_analysis, parsed_data):
        
        diagnosis_text = diagnosis.get("diagnosis", "")
        form_issues = form_analysis.get("form_analysis", "")
        web_results = research.get("web_results", [])
        
        prompt = f"""Create action plan (NO markdown ** symbols):

DIAGNOSIS: {diagnosis_text}
FORM: {form_issues}

Format:
ROOT CAUSE
[explanation]

IMMEDIATE ACTION
- [action 1]
- [action 2]

THIS WEEK
- [exercise 1: sets/reps]
- [exercise 2: sets/reps]

MONITOR
- [what to track]

SEE PROFESSIONAL IF
- [red flag 1]

NO ** stars. Plain text with bullets."""
        
        action_plan = self.call_llm(prompt, max_tokens=600)
        
        # FORCE add references
        if web_results:
            action_plan += "\n\n" + "="*60
            action_plan += "\n📚 REFERENCES\n"
            action_plan += "="*60 + "\n"
            for i, r in enumerate(web_results, 1):
                if r.get('url'):
                    action_plan += f"\n{i}. {r['title']}\n"
                    action_plan += f"   🔗 {r['url']}\n"
        
        self.log_action("create_plan", {"refs": len(web_results)})
        
        return {
            "agent": self.name,
            "action_plan": action_plan,
            "status": "success"
        }
