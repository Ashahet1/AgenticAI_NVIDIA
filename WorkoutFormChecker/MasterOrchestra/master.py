# master.py

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from Conversation_Manager import ConversationManager
from planner_agent import PlannerAgent
from parsing_agent import ParsingAgent
from form_analysis_agent import FormAnalysisAgent
from injury_diagnosis_agent import InjuryDiagnosisAgent
from research_agent import ResearchAgent
from prescription_agent import PrescriptionAgent
from simple_extractor import SimpleExtractor


class DynamicMasterOrchestrator:
    def __init__(self):
        logger.info("🎼 Initializing Dynamic Master Orchestrator...")
        
        self.agents = {
            "ParsingAgent": ParsingAgent(),
            "FormAnalysisAgent": FormAnalysisAgent(),
            "InjuryDiagnosisAgent": InjuryDiagnosisAgent(),
            "ResearchAgent": ResearchAgent(),
            "PrescriptionAgent": PrescriptionAgent()
        }
        from base_agent import BaseAgent
        self.llm_client = BaseAgent(
            name="LLMClient",
            role="general reasoning and question generation"
        )

        self.planner = PlannerAgent()
        self.conversation_manager = ConversationManager(llm_client=self.llm_client)
        self.simple_extractor = SimpleExtractor()
        
        self.state = {
            "collected_data": {},
            "agent_results": {},
            "last_agent": None,
            "workflow_complete": False
        }
        
        logger.info("✅ All agents initialized!")
    
    def process_user_message(self, user_message):
        logger.info(f"\n{'='*70}")
        logger.info(f"📨 Processing: {user_message[:80]}...")
        logger.info(f"{'='*70}")
        
        self.conversation_manager.add_message("user", user_message)
        
        # First message - simple extraction
        if len(self.conversation_manager.conversation_history) == 1:
            logger.info("🔍 Running simple extraction...")
            simple_data = self.simple_extractor.extract(user_message)
            
            for key, value in simple_data.items():
                if value != "unknown":
                    self.conversation_manager.update_collected_data({key: value})
                    self.state["collected_data"] = self.conversation_manager.collected_data
        
        # Follow-up answer
        if self._is_follow_up_answer():
            self._handle_follow_up_answer(user_message)
        
        max_iterations = 50  # ✅ Reduced from 100
        iteration = 0
        agents_run_count = 0
        max_agents_per_request = 10  # ✅ Emergency brake
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\n🔄 Iteration {iteration}")
            
            # ✅ EMERGENCY BRAKE: Stop if too many agents ran
            if agents_run_count >= max_agents_per_request:
                logger.error(f"🚨 EMERGENCY BRAKE: {agents_run_count} agents ran!")
                logger.error("   → Forcing completion")
                
                self.state["workflow_complete"] = True
                final_result = self._compile_final_result()
                
                return {
                    "type": "complete",
                    "result": final_result,
                    "conversation": self.conversation_manager.conversation_history,
                    "execution_summary": self.planner.get_execution_summary(),
                    "warning": f"Emergency brake at {agents_run_count} agents"
                }
            
            decision = self.planner.decide_next_action({
                **self.state,
                "conversation_turns": len(self.conversation_manager.conversation_history),
                "needs_clarification": self.conversation_manager.needs_clarification()
            })
            
            action = decision["action"]
            reason = decision["reason"]
            
            logger.info(f"   🧠 Planner: {action}")
            logger.info(f"   📝 Reason: {reason}")
            
            if action == "ask_user":
                question = decision.get("question")
                
                if question is None:
                    question = self.conversation_manager.generate_clarifying_question()
                
                if not question:
                    logger.info("   No more questions - continuing")
                    continue
                
                self.conversation_manager.add_message("agent", question, "PlannerAgent")
                
                return {
                    "type": "question",
                    "question": question,
                    "reason": reason,
                    "conversation": self.conversation_manager.conversation_history
                }
            
            elif action == "run_agent":
                agent_name = decision["agent"]
                
                # ✅ Safety check: Don't run same agent twice
                if self.state.get("last_agent") == agent_name:
                    logger.warning(f"   ⚠️ Trying to run {agent_name} again! Skipping.")
                    continue
                
                result = self._execute_agent(agent_name, user_message)
                
                self.state["agent_results"][agent_name] = result
                self.state["last_agent"] = agent_name
                
                if "parsed_data" in result:
                    self.conversation_manager.update_collected_data(result["parsed_data"])
                    self.state["collected_data"] = self.conversation_manager.collected_data
                
                self.planner.record_agent_execution(agent_name)
                agents_run_count += 1  # ✅ Track agent executions
                
                continue
            
            elif action == "complete":
                logger.info("\n✅ Workflow complete")
                self.state["workflow_complete"] = True
                final_result = self._compile_final_result()
                
                return {
                    "type": "complete",
                    "result": final_result,
                    "conversation": self.conversation_manager.conversation_history,
                    "execution_summary": self.planner.get_execution_summary()
                }
            
            else:
                logger.warning(f"⚠️ Unknown action: {action}")
                break
        
        logger.warning(f"⚠️ Max iterations reached ({max_iterations})")
        self.state["workflow_complete"] = True
        final_result = self._compile_final_result()
        
        return {
            "type": "complete",
            "result": final_result,
            "conversation": self.conversation_manager.conversation_history,
            "execution_summary": self.planner.get_execution_summary(),
            "warning": f"Reached max iterations ({max_iterations})"
        }
    
    def _is_follow_up_answer(self):
        history = self.conversation_manager.conversation_history
        if len(history) < 2:
            return False
        return history[-2].get("role") == "agent"
    
    def _handle_follow_up_answer(self, answer):
        """Capture user's answer to last question"""
        history = self.conversation_manager.conversation_history
        if len(history) < 2:
            return
        
        last_q = history[-2].get("content", "").lower()
        answer_lower = answer.lower().strip()
        
        # Map answer to field
        if "exercise" in last_q:
            self.conversation_manager.update_collected_data({"exercise": answer})
        elif "where" in last_q or "location" in last_q:
            self.conversation_manager.update_collected_data({"pain_location": answer})
        elif "when" in last_q or "timing" in last_q:
            self.conversation_manager.update_collected_data({"pain_timing": answer})
        elif "intense" in last_q or "severity" in last_q:
            self.conversation_manager.update_collected_data({"pain_intensity": answer})
        elif "type" in last_q or "describe" in last_q:
            self.conversation_manager.update_collected_data({"pain_type": answer})
        
        self.state["collected_data"] = self.conversation_manager.collected_data
        logger.info(f"✅ Captured answer: {answer[:50]}")
    
    def _execute_agent(self, agent_name, context):
        """Execute a specific agent"""
        logger.info(f"⚡ Executing: {agent_name}")

        agent = self.agents.get(agent_name)
        if not agent:
            logger.error(f"❌ Agent not found: {agent_name}")
            return {"error": f"Agent {agent_name} not found"}

        try:
            # --------------------------
            # 1️⃣ Parsing Agent
            # --------------------------
            if agent_name == "ParsingAgent":
                result = agent.execute(self.state["collected_data"])

            # --------------------------
            # 2️⃣ Form Analysis Agent
            # --------------------------
            elif agent_name == "FormAnalysisAgent":
                result = agent.execute(self.state["collected_data"])

            # --------------------------
            # 3️⃣ Injury Diagnosis Agent
            # --------------------------
            elif agent_name == "InjuryDiagnosisAgent":
                form_analysis = self.state["agent_results"].get("FormAnalysisAgent", {})
                form_dict = {"form_analysis": form_analysis.get("form_analysis", "")}
                result = agent.execute(self.state["collected_data"], form_dict)

            # --------------------------
            # 4️⃣ Research Agent
            # --------------------------
            elif agent_name == "ResearchAgent":
                diagnosis_result = self.state["agent_results"].get("InjuryDiagnosisAgent", {})
                diagnosis_dict = {"diagnosis": diagnosis_result.get("diagnosis", "unknown")}
                exercise_info = self.state["collected_data"]
                result = agent.execute(diagnosis_dict, exercise_info)

            # --------------------------
            # 5️⃣ Prescription Agent
            # --------------------------
            elif agent_name == "PrescriptionAgent":
                diagnosis_result = self.state["agent_results"].get("InjuryDiagnosisAgent", {})
                research_result = self.state["agent_results"].get("ResearchAgent", {})
                form_result = self.state["agent_results"].get("FormAnalysisAgent", {})
                parsed_result = self.state["agent_results"].get("ParsingAgent", {})

                diagnosis = {"diagnosis": diagnosis_result.get("diagnosis", "")}
                research = {
                    "web_results": research_result.get("web_results", []),
                    "findings": research_result.get("findings", "")
                }
                form_analysis = {"form_analysis": form_result.get("form_analysis", "")}
                parsed_data = parsed_result.get("parsed_data", self.state["collected_data"])

                result = agent.execute(diagnosis, research, form_analysis, parsed_data)

            # --------------------------
            # Default
            # --------------------------
            else:
                result = agent.execute(self.state["collected_data"])

            logger.info(f"✅ {agent_name} complete")
            return result

        except Exception as e:
            logger.error(f"❌ {agent_name} failed: {e}")
            return {"error": str(e), "agent": agent_name}

    
    def _compile_final_result(self):
        """Compile final result from all agents"""
        return {
            "parsed_data": self.state["agent_results"].get("ParsingAgent", {}).get("parsed_data", self.state["collected_data"]),
            "form_analysis": self.state["agent_results"].get("FormAnalysisAgent", {}).get("form_analysis", "N/A"),
            "diagnosis": self.state["agent_results"].get("InjuryDiagnosisAgent", {}).get("diagnosis", "N/A"),
            "research_findings": self.state["agent_results"].get("ResearchAgent", {}).get("findings", "N/A"),
            "action_plan": self.state["agent_results"].get("PrescriptionAgent", {}).get("action_plan", "N/A"),
            "web_sources": self.state["agent_results"].get("ResearchAgent", {}).get("sources", []),
            "agents_executed": self.planner.execution_history
        }
    
    def reset_conversation(self):
        self.conversation_manager = ConversationManager()
        self.planner = PlannerAgent()
        self.state = {
            "collected_data": {},
            "agent_results": {},
            "last_agent": None,
            "workflow_complete": False
        }
        logger.info("🔄 Conversation reset")

# ====================== TEST SECTION ======================
if __name__ == "__main__":
    print("🧪 Testing DynamicMasterOrchestrator...\n")

    orchestrator = DynamicMasterOrchestrator()

    # Simple user input simulation
    user_input = "I felt a sharp pain in my right knee while doing squats yesterday."

    print(f"\n➡️ User message: {user_input}")
    result = orchestrator.process_user_message(user_input)

    # Handle follow-up if it returns a question
    if result["type"] == "question":
        print(f"\n🤖 Question asked: {result['question']}")
        follow_up = "It happens during ascent when I stand up."
        print(f"🧍 User follow-up: {follow_up}")
        result = orchestrator.process_user_message(follow_up)
      # ✅ Add this line here
    orchestrator.conversation_manager.force_proceed()

    # Continue to diagnosis, research, and prescription
    result = orchestrator.process_user_message("ready to proceed")

    print("\n✅ Final result type:", result.get("type"))
    
    print("🧩 Final state summary:")
    print(result.get("result", {}))
    print("\n🧠 Agents executed:", result.get("execution_summary", []))
