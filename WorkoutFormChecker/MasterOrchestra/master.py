import os
import sys
import time
from io import StringIO
from contextlib import redirect_stdout

# Ensure imports work regardless of runtime path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from parsing_agent import ParsingAgent
from form_analysis_agent import FormAnalysisAgent
from injury_diagnosis_agent import InjuryDiagnosisAgent
from research_agent import ResearchAgent
from prescription_agent import PrescriptionAgent


# =====================================================
# 🧠 Reasoning Controller — the reflective agent brain
# =====================================================
class ReasoningController:
    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts
        self.reflections = []
        self.confidence_map = {"low": 1, "medium": 2, "high": 3}

    def evaluate_confidence(self, level):
        return self.confidence_map.get(str(level).lower(), 1)

    def reflect(self, step, agent_name, confidence, notes=""):
        """Logs human-like reflection and decides whether to retry."""
        print(f"\n🧠 Reflection after {agent_name}:")
        if confidence == "high":
            print(f"✨ Confidence is high. Reasoning stable — moving forward.")
            return False
        elif confidence == "medium":
            print(f"🤔 Moderate confidence detected. {notes or 'Continuing but noting possible uncertainty.'}")
            return False
        else:
            print(f"⚠️ Low confidence detected from {agent_name}. Triggering deeper reflection...")
            self.reflections.append(f"{agent_name} uncertainty — cause: {notes or 'unclear pattern'}")
            return True

    def summarize(self):
        print("\n" + "="*60)
        print("🪞 REFLECTION SUMMARY")
        print("="*60)
        if not self.reflections:
            print("✅ All reasoning steps reached stable confidence without retries.")
        else:
            for i, r in enumerate(self.reflections, 1):
                print(f"{i}. {r}")
        print("="*60 + "\n")


# =====================================================
# 🎼 Master Orchestrator (Agentic version)
# =====================================================
class MasterOrchestrator:

    def __init__(self):
        print("🎼 Initializing Master Orchestrator with reasoning loop...")
        self.parsing_agent = ParsingAgent()
        self.form_agent = FormAnalysisAgent()
        self.injury_agent = InjuryDiagnosisAgent()
        self.research_agent = ResearchAgent()
        self.prescription_agent = PrescriptionAgent()
        self.reasoning = ReasoningController()
        self.workflow_steps = []
        print("✅ All agents initialized successfully!")

    def log_step(self, step_name, agent_name, result):
        self.workflow_steps.append({
            "step": len(self.workflow_steps) + 1,
            "action": step_name,
            "agent": agent_name,
            "status": result.get("status", "unknown"),
            "confidence": result.get("confidence", "unknown")
        })

    # ================================================
    # 🔁 Agentic reasoning-driven workflow
    # ================================================
    def analyze_workout_issue(self, user_input):
        print("\n" + "="*60)
        print("🤖 AGENTIC MULTI-AGENT ANALYSIS STARTED")
        print("="*60)

        enriched_context = None
        attempts = 0

        # Reasoning loop
        while attempts < self.reasoning.max_attempts:
            attempts += 1
            print(f"\n🔄 Iteration {attempts}/{self.reasoning.max_attempts}")

            # STEP 1: Parsing
            print("\n📝 STEP 1: Parsing user input...")
            parsing_result = self.parsing_agent.execute(user_input)
            self.log_step("Parse Input", "ParsingAgent", parsing_result)
            parsed_data = parsing_result.get("parsed_data", {})

            # Merge reasoning fields if available
            parsed_data.update({
                "duration_since_onset": parsed_data.get("duration_since_onset", "unknown"),
                "previous_injuries": parsed_data.get("previous_injuries", "none reported"),
                "training_experience": parsed_data.get("training_experience", "unspecified"),
                "warmup_routine": parsed_data.get("warmup_routine", "unspecified"),
                "equipment": parsed_data.get("equipment", "not specified"),
                "surface": parsed_data.get("surface", "unknown"),
                "environment": parsed_data.get("environment", "unspecified"),
                "what_they_did_after": parsed_data.get("what_they_did_after", "no action taken"),
                "improvement_since": parsed_data.get("improvement_since", "unknown"),
                "goal": parsed_data.get("goal", "recover and continue training safely")
            })

            print(f"✅ Extracted: {parsed_data.get('exercise')} | {parsed_data.get('pain_location')} | {parsed_data.get('pain_timing')}")
            print(f"🧩 Extended reasoning fields integrated for deeper context.")

            # STEP 2: Form analysis
            print("\n🏋️ STEP 2: Analyzing form patterns and deviations...")
            form_result = self.form_agent.execute(parsed_data)
            self.log_step("Analyze Form", "FormAnalysisAgent", form_result)
            print(f"🧠 Form Analysis Insight: {form_result.get('form_analysis', 'N/A')}")
            print(f"🧾 Confidence: {form_result.get('confidence')}")

            if not self.reasoning.reflect("FormAnalysis", "FormAnalysisAgent", form_result.get("confidence", "low")):
                pass
            else:
                print("🔁 Re-evaluating with adjusted biomechanics assumptions...\n")
                continue

            # STEP 3: Injury diagnosis
            print("\n🩺 STEP 3: Diagnosing injury pattern based on symptoms and form...")
            injury_result = self.injury_agent.execute(parsed_data, form_result)
            self.log_step("Diagnose Injury", "InjuryDiagnosisAgent", injury_result)
            print(f"🧠 Injury Diagnosis: {injury_result.get('diagnosis')}")
            print(f"📉 Confidence: {injury_result.get('confidence')}")

            if not self.reasoning.reflect("InjuryDiagnosis", "InjuryDiagnosisAgent", injury_result.get("confidence", "low")):
                pass
            else:
                print("🔁 Re-thinking injury mechanism and context...")
                continue

            # STEP 4: Research
            print("\n🔍 STEP 4: Researching relevant evidence and studies...")
            research_result = self.research_agent.execute(injury_result, parsed_data)
            self.log_step("Research Evidence", "ResearchAgent", research_result)
            print(f"📚 Found {len(research_result.get('web_results', []))} supporting studies.")
            print(f"🧠 Research synthesis: {research_result.get('synthesis', 'N/A')}")

            # STEP 5: Prescription
            print("\n📋 STEP 5: Building personalized recovery plan...")
            prescription_result = self.prescription_agent.execute(
                injury_result,
                research_result,
                form_result,
                parsed_data
            )
            self.log_step("Create Action Plan", "PrescriptionAgent", prescription_result)
            print(f"✅ Plan Generated: {prescription_result.get('action_plan', '')[:200]}...")

            # Confidence check
            final_conf = injury_result.get("confidence", "low")
            if self.reasoning.evaluate_confidence(final_conf) >= 2:
                print("\n🎯 Confidence stabilized — exiting reasoning loop.")
                break
            else:
                print("\n🔁 Confidence still low — triggering self-reflection cycle.")
                time.sleep(1.5)

        print("\n" + "="*60)
        print("✅ AGENTIC MULTI-AGENT ANALYSIS COMPLETE")
        print("="*60)

        # Final reflection summary
        #self.reasoning.summarize()

        return {
            "user_input": user_input,
            "parsed_data": parsed_data,
            "form_analysis": form_result.get("form_analysis"),
            "diagnosis": injury_result.get("diagnosis"),
            "research_findings": research_result.get("synthesis"),
            "web_sources": research_result.get("sources", []),
            "web_results": research_result.get("web_results", []),
            "action_plan": prescription_result.get("action_plan"),
            "workflow_steps": self.workflow_steps,
            "total_steps": len(self.workflow_steps),
            "requires_medical_attention": injury_result.get("requires_medical_attention", False)
        }


# =====================================================
# Entry Points
# =====================================================
def run(user_input: str):
    """Entry for server or Colab."""
    buf = StringIO()
    orch = MasterOrchestrator()
    with redirect_stdout(buf):
        result = orch.analyze_workout_issue(user_input)

        print("\n" + "="*60)
        print("📊 FINAL RESULTS")
        print("="*60)
        print("\n🎯 ACTION PLAN:")
        print(result["action_plan"])

    printed_output = buf.getvalue()
    return {"result": result, "printed": printed_output}


def main(argv=None):
    argv = argv or []
    if argv:
        user_input = " ".join(argv)
    else:
        user_input = "Deadlift — sharp lower back pain during lift, started 3 days ago."
    return run(user_input)

# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    print("🧪 Running Agentic MasterOrchestrator in standalone mode...\n")

    test_input = "I feel sharp shoulder pain"
    output = run(test_input)

    # Just print once — the captured version
    print(output["printed"])
    

