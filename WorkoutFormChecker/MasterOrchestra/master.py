import os
import sys
from contextlib import redirect_stdout

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from parsing_agent import ParsingAgent
from form_analysis_agent import FormAnalysisAgent
from injury_diagnosis_agent import InjuryDiagnosisAgent
from research_agent import ResearchAgent
from prescription_agent import PrescriptionAgent

class MasterOrchestrator:
    
    def __init__(self):
        print("🎼 Initializing Master Orchestrator...")
        
        self.parsing_agent = ParsingAgent()
        self.form_agent = FormAnalysisAgent()
        self.injury_agent = InjuryDiagnosisAgent()
        self.research_agent = ResearchAgent()
        self.prescription_agent = PrescriptionAgent()
        
        self.workflow_steps = []
        
        print("✅ All agents initialized!")
    
    def log_step(self, step_name, agent_name, result):
        self.workflow_steps.append({
            "step": len(self.workflow_steps) + 1,
            "action": step_name,
            "agent": agent_name,
            "status": result.get("status", "unknown")
        })
    
    def analyze_workout_issue(self, user_input):
        
        print("\n" + "="*60)
        print("🤖 MULTI-AGENT ANALYSIS STARTED")
        print("="*60)
        
        # STEP 1
        print("\n📝 STEP 1: Parsing user input...")
        parsing_result = self.parsing_agent.execute(user_input)
        self.log_step("Parse Input", "ParsingAgent", parsing_result)
        parsed_data = parsing_result['parsed_data']
        print(f"✅ Extracted: {parsed_data.get('exercise')} | {parsed_data.get('pain_location')} | {parsed_data.get('pain_timing')}")
        
        # STEP 2
        print("\n🏋️ STEP 2: Analyzing exercise form...")
        form_result = self.form_agent.execute(parsed_data)
        self.log_step("Analyze Form", "FormAnalysisAgent", form_result)
        print(f"✅ Form analysis complete (confidence: {form_result.get('confidence')})")
        
        # STEP 3
        print("\n🩺 STEP 3: Diagnosing injury pattern...")
        injury_result = self.injury_agent.execute(parsed_data, form_result)
        self.log_step("Diagnose Injury", "InjuryDiagnosisAgent", injury_result)
        print(f"✅ Diagnosis complete (confidence: {injury_result.get('confidence')})")
        
        
        # STEP 4
        print("\n🔍 STEP 4: Researching supporting evidence...")
        research_result = self.research_agent.execute(injury_result, parsed_data)
        self.log_step("Research Evidence", "ResearchAgent", research_result)
        print(f"✅ Found {len(research_result.get('web_results', []))} web sources")
        
        # STEP 5
        print("\n📋 STEP 5: Creating personalized action plan...")
        prescription_result = self.prescription_agent.execute(
            injury_result, 
            research_result, 
            form_result, 
            parsed_data
        )
        self.log_step("Create Action Plan", "PrescriptionAgent", prescription_result)
        print("✅ Action plan generated!")
        
        print("\n" + "="*60)
        print("✅ MULTI-AGENT ANALYSIS COMPLETE")
        print("="*60)
        
        final_result = {
            "user_input": user_input,
            "parsed_data": parsed_data,
            "form_analysis": form_result.get('form_analysis'),
            "diagnosis": injury_result.get('diagnosis'),
            "research_findings": research_result.get('synthesis'),
            "web_sources": research_result.get('sources', []),
            "web_results": research_result.get('web_results', []),
            "action_plan": prescription_result.get('action_plan'),
            "workflow_steps": self.workflow_steps,
            "total_steps": len(self.workflow_steps),
            "requires_medical_attention": injury_result.get('requires_medical_attention', False)
        }
        
        return final_result
    
    def get_workflow_summary(self):
        return {
            "total_steps": len(self.workflow_steps),
            "steps": self.workflow_steps
        }

# ============ MODULE-LEVEL FUNCTIONS (Outside the class) ============

def run(user_input: str):
    """
    Import-friendly entrypoint for server usage.
    Returns both the final_result dict and the printed output text.
    """
    from io import StringIO
    from contextlib import redirect_stdout

    buf = StringIO()
    orch = MasterOrchestrator()
    with redirect_stdout(buf):
        result = orch.analyze_workout_issue(user_input)

        # Replicate the pretty summary block
        print("\n" + "="*60)
        print("📊 FINAL RESULTS")
        print("="*60)

        print("\n🎯 ACTION PLAN:")
        print(result['action_plan'])

        
    printed_output = buf.getvalue()
    return {"result": result, "printed": printed_output}


def main(argv=None):
    """
    Main function that accepts command-line args
    """
    argv = argv or []
    
    if argv:
        user_input = " ".join(argv)
    else:
        user_input = "I did squats today and my right knee hurts when standing up"
    
    return run(user_input)

# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    
    print("🧪 Testing Master Orchestrator with MULTIPLE test cases...\n")
    
    # TEST CASES
    test_cases = [
        "My lower back hurts after deadlifts, especially at the bottom of the movement",
        "I feel sharp shoulder pain when lowering the bar during bench press",
        "I did squats today and my right knee hurts when standing up"
    ]
    
    # Run test case (change index to test different cases)
    test_input = test_cases[1]  # 0=deadlift, 1=bench, 2=squat
    
    print(f"📝 Testing with: '{test_input}'\n")
    
    result = run(test_input)  # Uses module-level run() function
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print("\n🎯 ACTION PLAN:")
    print(result['action_plan'])
    
    #print("\n📚 WORKFLOW SUMMARY:")
    #for step in result['workflow_steps']:
    #    print(f"  {step['step']}. {step['action']} - {step['agent']} [{step['status']}]")
    
    #print("\n🔗 WEB SOURCES FOUND:")
    #for i, result_item in enumerate(result.get('web_results', []), 1):
    #    print(f"  {i}. {result_item.get('title', 'N/A')}")
    #    print(f"     {result_item.get('url', 'N/A')}")
