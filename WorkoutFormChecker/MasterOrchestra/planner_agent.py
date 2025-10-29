# planner_agent.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_agent import BaseAgent

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="PlannerAgent", role="orchestrating dynamic workflow")
        self.execution_history = []
    
    def decide_next_action(self, state):
      """
      Use NVIDIA Llama 3.1 (Nemotron) to dynamically decide which agent to run next.
      Falls back to static sequence if model call fails.
      """
      from openai import OpenAI
      import json, re, os
      from dotenv import load_dotenv

      # --- Load environment keys and init client ---
      load_dotenv('/content/drive/MyDrive/WorkoutFormChecker/.env')
      NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

      client = OpenAI(
          base_url="https://integrate.api.nvidia.com/v1",
          api_key=NVIDIA_API_KEY
      )

      # --- Extract conversation context ---
      collected_data = state.get("collected_data", {})
      conversation_turns = state.get("conversation_turns", 0)
      last_agent = state.get("last_agent", None)
      user_message = state.get("user_message", "")
      execution_history = getattr(self, "execution_history", [])

      print(f"\n🧠 Planner analyzing (via NVIDIA Llama 3.1)...")
      print(f"   Agents run: {execution_history}")
      print(f"   Turns: {conversation_turns}")
      print(f"   Collected data: {list(collected_data.keys())}")

      # --- Build reasoning context for the model ---
      context_summary = (
          f"Executed agents: {execution_history}\n"
          f"Last agent: {last_agent}\n"
          f"Collected data keys: {list(collected_data.keys())}\n"
          f"Conversation turns: {conversation_turns}\n"
          f"User message: {user_message[:250]}"
      )

      prompt = f"""
  You are the planner for a multi-agent rehabilitation assistant.

  Context:
  {context_summary}

  Available agents:
  1. ParsingAgent – extract structured info
  2. FormAnalysisAgent – analyze form & movement
  3. InjuryDiagnosisAgent – infer likely cause
  4. ResearchAgent – search KB/web for evidence
  5. PrescriptionAgent – generate rehab plan

  Return only JSON:
  {{"next_agent": "FormAnalysisAgent", "reason": "Because user described pain during movement"}}.
  """

      try:
          # --- Call NVIDIA Llama 3.1 reasoning model ---
          completion = client.chat.completions.create(
              model="nvidia/llama-3.1-nemotron-nano-8b-v1",
              messages=[{"role": "user", "content": prompt}],
              temperature=0.2,
              top_p=0.95,
              max_tokens=512,
          )

          raw_text = completion.choices[0].message.content
          match = re.search(r"\{.*\}", raw_text, re.DOTALL)
          decision = json.loads(match.group(0)) if match else {}

          next_agent = decision.get("next_agent")
          reason = decision.get("reason", "No reason provided")

          if next_agent not in [
              "ParsingAgent",
              "FormAnalysisAgent",
              "InjuryDiagnosisAgent",
              "ResearchAgent",
              "PrescriptionAgent",
          ]:
              raise ValueError("Invalid or missing next_agent in model response")

          print(f"🧠 NVIDIA Planner decided → {next_agent} ({reason})")
          return {"action": "run_agent", "agent": next_agent, "reason": reason}

      except Exception as e:
          print(f"⚠️ Planner reasoning failed ({e}) — reverting to static logic")

          # --- Static fallback sequence ---
          if "ParsingAgent" not in execution_history:
              return {"action": "run_agent", "agent": "ParsingAgent", "reason": "Fallback: parse input"}
          elif "FormAnalysisAgent" not in execution_history:
              return {"action": "run_agent", "agent": "FormAnalysisAgent", "reason": "Fallback: analyze form"}
          elif "InjuryDiagnosisAgent" not in execution_history:
              return {"action": "run_agent", "agent": "InjuryDiagnosisAgent", "reason": "Fallback: diagnose injury"}
          elif "ResearchAgent" not in execution_history:
              return {"action": "run_agent", "agent": "ResearchAgent", "reason": "Fallback: research evidence"}
          elif "PrescriptionAgent" not in execution_history:
              return {"action": "run_agent", "agent": "PrescriptionAgent", "reason": "Fallback: generate recovery plan"}
          else:
              return {"action": "complete", "reason": "All agents executed"}

    
    def _generate_question(self, collected_data):
        """Generate question for missing REQUIRED fields only"""
        if not collected_data.get("exercise") or collected_data.get("exercise") in ["unknown", "unspecified"]:
            return "What exercise were you doing when this happened?"
        
        if not collected_data.get("pain_location") or collected_data.get("pain_location") in ["unknown", "unspecified"]:
            return "Where exactly do you feel the pain? (e.g., right knee, left shoulder, lower back)"
        
        if not collected_data.get("pain_timing") or collected_data.get("pain_timing") in ["unknown", "unspecified"]:
            return "When does the pain occur? During the movement, after the workout, or the next day?"
        
        return "Can you provide more details about your injury?"
    
    def _generate_optional_question(self):
        """This will be overridden by conversation_manager's logic"""
        return "Let me gather some additional details..."
    
    def _needs_form_analysis(self, collected_data):
        """Decide if form analysis is needed"""
        pain_timing = collected_data.get("pain_timing", "").lower()
        
        # Skip if pain is after workout
        if any(x in pain_timing for x in ["after", "next day", "later", "post"]):
            return False
        
        # Need form analysis if pain during movement
        if any(x in pain_timing for x in ["during", "ascent", "descent", "bottom", "top", "lockout", "lowering", "pressing", "coming up", "going down"]):
            return True
        
        return True
    
    def record_agent_execution(self, agent_name):
        self.execution_history.append(agent_name)
        self.log_action("agent_executed", {"agent": agent_name})
    
    def get_execution_summary(self):
        return {
            "total_agents_run": len(self.execution_history),
            "agents_executed": self.execution_history,
            "unique_agents": list(set(self.execution_history))
        }