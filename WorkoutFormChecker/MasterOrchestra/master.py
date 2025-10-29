# master.py

import os
import sys
import time
from io import StringIO
from contextlib import redirect_stdout

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from conversation_manager import ConversationManager
from planner_agent import PlannerAgent
from parsing_agent import ParsingAgent
from form_analysis_agent import FormAnalysisAgent
from injury_diagnosis_agent import InjuryDiagnosisAgent
from research_agent import ResearchAgent
from prescription_agent import PrescriptionAgent
from simple_extractor import SimpleExtractor


class ReflectionController:
    def __init__(self):
        self.reflections = []
        self.confidence_map = {"low": 1, "medium": 2, "high": 3}
    
    def evaluate_confidence(self, level):
        return self.confidence_map.get(str(level).lower(), 1)
    
    def reflect_on_result(self, agent_name, result, state):
        confidence = result.get("confidence", "unknown")
        print(f"\n🪞 Reflection on {agent_name}: Confidence: {confidence}")
        
        if confidence == "high":
            return {"should_continue": True, "needs_more_info": False, "feedback": "High quality result"}
        elif confidence == "medium":
            return {"should_continue": True, "needs_more_info": False, "feedback": "Acceptable confidence level"}
        else:
            self.reflections.append(f"{agent_name}: Low confidence detected")
            return {"should_continue": True, "needs_more_info": True, "feedback": "Low confidence"}
    
    def get_reflection_summary(self):
        return {"total_reflections": len(self.reflections), "issues": self.reflections}


class DynamicMasterOrchestrator:
    def __init__(self):
        print("🎼 Initializing Dynamic Master Orchestrator...")
        
        self.agents = {
            "ParsingAgent": ParsingAgent(),
            "FormAnalysisAgent": FormAnalysisAgent(),
            "InjuryDiagnosisAgent": InjuryDiagnosisAgent(),
            "ResearchAgent": ResearchAgent(),
            "PrescriptionAgent": PrescriptionAgent()
        }
        
        self.planner = PlannerAgent()
        self.conversation_manager = ConversationManager()
        self.reflection = ReflectionController()
        self.simple_extractor = SimpleExtractor()
        
        self.state = {
            "collected_data": {},
            "agent_results": {},
            "confidence": "unknown",
            "last_agent": None,
            "workflow_complete": False
        }
        
        print("✅ All agents initialized!")
    
    def process_user_message(self, user_message):
        print("\n" + "="*70)
        print(f"📨 Processing: {user_message[:80]}...")
        print("="*70)
        
        self.conversation_manager.add_message("user", user_message)
        
        # FIRST: Try simple extraction on first message
        if len(self.conversation_manager.conversation_history) == 1:
            print("\n🔍 Running simple extraction on initial message...")
            simple_data = self.simple_extractor.extract(user_message)
            
            # Only use extracted data if it's not "unknown"
            for key, value in simple_data.items():
                if value != "unknown":
                    self.conversation_manager.update_collected_data({key: value})
                    self.state["collected_data"] = self.conversation_manager.collected_data
            
            print(f"   Simple extraction found: {list(simple_data.keys())}")
            print(f"   Collected data now: {self.state['collected_data']}")
        
        # Check if follow-up answer
        if self._is_follow_up_answer():
            self._handle_follow_up_answer(user_message)
        
        max_iterations = 100  # Increased to handle extensive Q&A
        iteration = 0
        last_progress_check = 0
        last_agent_count = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}")
            
            # Every 5 iterations, check if we're making progress
            if iteration % 5 == 0:
                current_agent_count = len(self.planner.execution_history)
                print(f"\n🔍 Progress check at iteration {iteration}:")
                print(f"   Agents run: {current_agent_count} (last check: {last_agent_count})")
                
                if current_agent_count == last_agent_count:
                    print(f"   ⚠️ STUCK! No agent progress in last 5 iterations")
                    print(f"   → Forcing conversation to complete and proceed to diagnosis")
                    
                    # Force conversation manager to stop asking questions
                    self.conversation_manager.force_proceed()
                    
                    # If we're still before diagnosis, force it to run
                    if "InjuryDiagnosisAgent" not in self.planner.execution_history:
                        print(f"   → Forcing InjuryDiagnosisAgent to run")
                        result = self._execute_agent("InjuryDiagnosisAgent", "")
                        self.state["agent_results"]["InjuryDiagnosisAgent"] = result
                        self.planner.record_agent_execution("InjuryDiagnosisAgent")
                else:
                    print(f"   ✓ Making progress! {current_agent_count - last_agent_count} agents ran")
                
                last_agent_count = current_agent_count
            
            decision = self.planner.decide_next_action({
                **self.state,
                "conversation_turns": len(self.conversation_manager.conversation_history)
            })
            
            action = decision["action"]
            reason = decision["reason"]
            
            print(f"   🧠 Planner: {action}")
            print(f"   📝 Reason: {reason}")
            
            if action == "ask_user":
                question = decision["question"]
                
                # If planner didn't provide a question, get one from conversation_manager
                if question is None:
                    question = self.conversation_manager.generate_clarifying_question()
                
                if not question:
                    # No more questions to ask, proceed
                    print("   No more questions - proceeding to diagnosis")
                    continue
                
                self.conversation_manager.add_message("agent", question, "PlannerAgent")
                
                return {
                    "type": "question",
                    "question": question,
                    "reason": reason,
                    "conversation": self.conversation_manager.conversation_history,
                    "state": self._get_state_snapshot()
                }
            
            elif action == "run_agent":
                agent_name = decision["agent"]
                result = self._execute_agent(agent_name, user_message)
                
                self.state["agent_results"][agent_name] = result
                self.state["last_agent"] = agent_name
                
                if "parsed_data" in result:
                    self.conversation_manager.update_collected_data(result["parsed_data"])
                    self.state["collected_data"] = self.conversation_manager.collected_data
                
                if "confidence" in result:
                    self.state["confidence"] = result["confidence"]
                
                self.planner.record_agent_execution(agent_name)
                self.reflection.reflect_on_result(agent_name, result, self.state)
                
                continue
            
            elif action == "complete":
                print("\n✅ Workflow complete")
                self.state["workflow_complete"] = True
                final_result = self._compile_final_result()
                
                return {
                    "type": "complete",
                    "result": final_result,
                    "conversation": self.conversation_manager.conversation_history,
                    "execution_summary": self.planner.get_execution_summary(),
                    "reflection_summary": self.reflection.get_reflection_summary(),
                    "state": self._get_state_snapshot()
                }
            
            else:
                print(f"⚠️ Unknown action: {action}")
                break
        
        print(f"⚠️ Max iterations reached ({max_iterations}) - forcing completion")
        
        # Force complete with whatever we have
        self.state["workflow_complete"] = True
        final_result = self._compile_final_result()
        
        return {
            "type": "complete",
            "result": final_result,
            "conversation": self.conversation_manager.conversation_history,
            "execution_summary": self.planner.get_execution_summary(),
            "reflection_summary": self.reflection.get_reflection_summary(),
            "state": self._get_state_snapshot(),
            "warning": f"Reached max iterations ({max_iterations}), completing with available data"
        }
    
    def _execute_agent(self, agent_name, context):
        print(f"\n⚡ Executing: {agent_name}")
        agent = self.agents[agent_name]
        
        if agent_name == "ParsingAgent":
            user_messages = [msg["content"] for msg in self.conversation_manager.conversation_history if msg.get("role") == "user"]
            full_context = " | ".join(user_messages)
            return agent.execute(full_context)
        
        elif agent_name == "FormAnalysisAgent":
            return agent.execute(self.state["collected_data"])
        
        elif agent_name == "InjuryDiagnosisAgent":
            form_result = self.state["agent_results"].get("FormAnalysisAgent", {})
            return agent.execute(self.state["collected_data"], form_result)
        
        elif agent_name == "ResearchAgent":
            diagnosis = self.state["agent_results"].get("InjuryDiagnosisAgent", {})
            return agent.execute(diagnosis, self.state["collected_data"])
        
        elif agent_name == "PrescriptionAgent":
            diagnosis = self.state["agent_results"].get("InjuryDiagnosisAgent", {})
            research = self.state["agent_results"].get("ResearchAgent", {})
            form = self.state["agent_results"].get("FormAnalysisAgent", {})
            return agent.execute(diagnosis, research, form, self.state["collected_data"])
        
        else:
            return {"error": f"Unknown agent: {agent_name}"}
    
    def _compile_final_result(self):
        agent_results = self.state["agent_results"]
        parsed_data = self.state.get("collected_data", {})
        
        form_analysis = agent_results.get("FormAnalysisAgent", {}).get("form_analysis", "N/A")
        diagnosis = agent_results.get("InjuryDiagnosisAgent", {}).get("diagnosis", "N/A")
        
        research = agent_results.get("ResearchAgent", {})
        research_synthesis = research.get("synthesis", "N/A")
        web_sources = research.get("sources", [])
        web_results = research.get("web_results", [])
        
        print(f"DEBUG _compile_final_result - web_results: {len(web_results)}, web_sources: {len(web_sources)}")
        
        action_plan = agent_results.get("PrescriptionAgent", {}).get("action_plan", "N/A")
        requires_medical = agent_results.get("InjuryDiagnosisAgent", {}).get("requires_medical_attention", False)
        
        return {
            "user_input": self.conversation_manager.conversation_history[0]["content"] if self.conversation_manager.conversation_history else "",
            "parsed_data": parsed_data,
            "form_analysis": form_analysis,
            "diagnosis": diagnosis,
            "research_findings": research_synthesis,
            "web_sources": web_sources,
            "web_results": web_results,
            "action_plan": action_plan,
            "agents_executed": self.planner.execution_history,
            "total_agents": len(set(self.planner.execution_history)),
            "conversation_turns": len(self.conversation_manager.conversation_history),
            "requires_medical_attention": requires_medical
        }
    
    def _get_state_snapshot(self):
        return {
            "collected_data": self.state["collected_data"],
            "confidence": self.state["confidence"],
            "last_agent": self.state["last_agent"],
            "agents_executed": self.planner.execution_history,
            "conversation_turns": len(self.conversation_manager.conversation_history)
        }
    
    def _is_follow_up_answer(self):
        history = self.conversation_manager.conversation_history
        if len(history) < 2:
            return False
        last_msg = history[-2]
        return last_msg.get("role") == "agent" and "?" in last_msg.get("content", "")
    
    def _handle_follow_up_answer(self, answer):
        history = self.conversation_manager.conversation_history
        if len(history) < 2:
            return
        
        last_agent_msg = history[-2].get("content", "").lower()
        answer_lower = answer.lower().strip()
        
        print(f"\n📝 Capturing follow-up answer:")
        print(f"   Question was: {last_agent_msg[:60]}...")
        print(f"   Answer: {answer_lower}")
        
        if "exercise" in last_agent_msg or "what exercise" in last_agent_msg or "doing when" in last_agent_msg:
            self.conversation_manager.update_collected_data({"exercise": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: exercise = {answer_lower}")
        elif ("where" in last_agent_msg and "pain" in last_agent_msg) or "pain location" in last_agent_msg:
            self.conversation_manager.update_collected_data({"pain_location": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: pain_location = {answer_lower}")
        elif ("when" in last_agent_msg and "pain" in last_agent_msg) or "pain occur" in last_agent_msg:
            self.conversation_manager.update_collected_data({"pain_timing": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: pain_timing = {answer_lower}")
        elif "which side" in last_agent_msg or ("left" in last_agent_msg and "right" in last_agent_msg):
            self.conversation_manager.update_collected_data({"pain_side": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: pain_side = {answer_lower}")
        elif "how intense" in last_agent_msg or "intensity" in last_agent_msg:
            self.conversation_manager.update_collected_data({"pain_intensity": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: pain_intensity = {answer_lower}")
        elif "what type" in last_agent_msg or "describe the pain" in last_agent_msg:
            self.conversation_manager.update_collected_data({"pain_type": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: pain_type = {answer_lower}")
        elif "movement phase" in last_agent_msg or "point in the movement" in last_agent_msg:
            self.conversation_manager.update_collected_data({"movement_phase": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: movement_phase = {answer_lower}")
        elif "how long ago" in last_agent_msg or "when did" in last_agent_msg or "duration" in last_agent_msg:
            self.conversation_manager.update_collected_data({"duration_since_onset": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: duration_since_onset = {answer_lower}")
        elif "similar" in last_agent_msg and "before" in last_agent_msg:
            self.conversation_manager.update_collected_data({"previous_injuries": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: previous_injuries = {answer_lower}")
        elif "training experience" in last_agent_msg or "how long" in last_agent_msg and "training" in last_agent_msg:
            self.conversation_manager.update_collected_data({"training_experience": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: training_experience = {answer_lower}")
        elif "equipment" in last_agent_msg or "what equipment" in last_agent_msg:
            self.conversation_manager.update_collected_data({"equipment": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: equipment = {answer_lower}")
        elif "sleep" in last_agent_msg:
            self.conversation_manager.update_collected_data({"sleep_quality": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: sleep_quality = {answer_lower}")
        elif "hydrat" in last_agent_msg:
            self.conversation_manager.update_collected_data({"hydration_level": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: hydration_level = {answer_lower}")
        elif "how often" in last_agent_msg or "training frequency" in last_agent_msg:
            self.conversation_manager.update_collected_data({"training_frequency": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: training_frequency = {answer_lower}")
        elif "other symptoms" in last_agent_msg or "associated symptoms" in last_agent_msg:
            self.conversation_manager.update_collected_data({"associated_symptoms": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: associated_symptoms = {answer_lower}")
        elif "tried" in last_agent_msg and "treat" in last_agent_msg:
            self.conversation_manager.update_collected_data({"self_treatment_actions": answer_lower})
            self.state["collected_data"] = self.conversation_manager.collected_data
            print(f"   ✓ Captured: self_treatment_actions = {answer_lower}")
        else:
            print(f"   ⚠️ No field mapping found for question: {last_agent_msg[:50]}")
