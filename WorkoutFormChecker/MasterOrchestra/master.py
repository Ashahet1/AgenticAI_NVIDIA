# master.py

import os
import sys
import time
from io import StringIO
from contextlib import redirect_stdout

# Ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from conversation_manager import ConversationManager
from planner_agent import PlannerAgent
from parsing_agent import ParsingAgent
from form_analysis_agent import FormAnalysisAgent
from injury_diagnosis_agent import InjuryDiagnosisAgent
from research_agent import ResearchAgent
from prescription_agent import PrescriptionAgent


# =====================================================
# 🪞 Reflection Controller (Quality Checks)
# =====================================================
class ReflectionController:
    """
    Evaluates agent output quality and confidence
    Decides if we need to retry, gather more info, or proceed
    """
    
    def __init__(self):
        self.reflections = []
        self.confidence_map = {"low": 1, "medium": 2, "high": 3}
    
    def evaluate_confidence(self, level):
        """Convert confidence string to numeric score"""
        return self.confidence_map.get(str(level).lower(), 1)
    
    def reflect_on_result(self, agent_name, result, state):
        """
        Evaluate agent result quality
        
        Returns:
            dict with:
                - should_continue: bool
                - needs_more_info: bool  
                - feedback: str
        """
        confidence = result.get("confidence", "unknown")
        
        print(f"\n🪞 Reflection on {agent_name}:")
        print(f"   Confidence: {confidence}")
        
        # Check confidence level
        if confidence == "high":
            print(f"   ✅ High confidence - proceeding")
            return {
                "should_continue": True,
                "needs_more_info": False,
                "feedback": "High quality result"
            }
        
        elif confidence == "medium":
            print(f"   🤔 Medium confidence - acceptable, continuing")
            return {
                "should_continue": True,
                "needs_more_info": False,
                "feedback": "Acceptable confidence level"
            }
        
        else:  # low or unknown
            print(f"   ⚠️ Low confidence - may need more evidence")
            self.reflections.append(f"{agent_name}: Low confidence detected")
            
            return {
                "should_continue": True,  # Let planner decide what to do
                "needs_more_info": True,
                "feedback": "Low confidence - consider gathering more evidence"
            }
    
    def get_reflection_summary(self):
        """Get summary of all reflections"""
        return {
            "total_reflections": len(self.reflections),
            "issues": self.reflections
        }


# =====================================================
# 🎼 Dynamic Master Orchestrator
# =====================================================
class DynamicMasterOrchestrator:
    """
    Dynamic multi-agent orchestration with conversational flow
    
    - Uses PlannerAgent to decide which agent runs next
    - Supports multi-turn conversation
    - Can pause and ask user for clarification
    - Dynamically adapts workflow based on case complexity
    """
    
    def __init__(self):
        print("🎼 Initializing Dynamic Master Orchestrator...")
        
        # Initialize all agents (pool, not pipeline!)
        self.agents = {
            "ParsingAgent": ParsingAgent(),
            "FormAnalysisAgent": FormAnalysisAgent(),
            "InjuryDiagnosisAgent": InjuryDiagnosisAgent(),
            "ResearchAgent": ResearchAgent(),
            "PrescriptionAgent": PrescriptionAgent()
        }
        
        # Initialize orchestration components
        self.planner = PlannerAgent()
        self.conversation_manager = ConversationManager()
        self.reflection = ReflectionController()
        
        # State tracking
        self.state = {
            "collected_data": {},
            "agent_results": {},
            "confidence": "unknown",
            "last_agent": None,
            "workflow_complete": False
        }
        
        print("✅ All agents and orchestration components initialized!")
    
    def process_user_message(self, user_message):
        """
        Process ONE user message in the dynamic workflow
        
        This is called each time user sends a message
        Can return:
        - "question" type: Need more info from user
        - "processing" type: Analysis in progress  
        - "complete" type: Final result ready
        
        Args:
            user_message: str - User's message
            
        Returns:
            dict with type, content, and state
        """
        
        print("\n" + "="*70)
        print(f"📨 Processing user message: {user_message[:80]}...")
        print("="*70)
        
        # Add user message to conversation
        self.conversation_manager.add_message("user", user_message)
        
        # Dynamic execution loop
        max_iterations = 15  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}")
            
            # STEP 1: Ask Planner what to do next
            decision = self.planner.decide_next_action(self.state)
            
            action = decision["action"]
            reason = decision["reason"]
            
            print(f"   🧠 Planner Decision: {action}")
            print(f"   📝 Reason: {reason}")
            
            # STEP 2: Execute decision
            
            # CASE A: Need to ask user a question
            if action == "ask_user":
                question = decision["question"]
                
                self.conversation_manager.add_message("agent", question, "PlannerAgent")
                
                return {
                    "type": "question",
                    "question": question,
                    "reason": reason,
                    "conversation": self.conversation_manager.conversation_history,
                    "state": self._get_state_snapshot()
                }
            
            # CASE B: Run an agent
            elif action == "run_agent":
                agent_name = decision["agent"]
                
                # Execute the agent
                result = self._execute_agent(agent_name, user_message)
                
                # Store result
                self.state["agent_results"][agent_name] = result
                self.state["last_agent"] = agent_name
                
                # Update collected data if parsing result
                if "parsed_data" in result:
                    self.conversation_manager.update_collected_data(result["parsed_data"])
                    self.state["collected_data"] = self.conversation_manager.collected_data
                
                # Update confidence
                if "confidence" in result:
                    self.state["confidence"] = result["confidence"]
                
                # Record agent execution
                self.planner.record_agent_execution(agent_name)
                
                # Reflection: Check result quality
                reflection_result = self.reflection.reflect_on_result(
                    agent_name, 
                    result, 
                    self.state
                )
                
                # Continue loop (planner will decide next action)
                continue
            
            # CASE C: Workflow complete
            elif action == "complete":
                print("\n✅ Workflow complete - compiling final result")
                
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
            
            # CASE D: Unknown action (shouldn't happen)
            else:
                print(f"⚠️ Unknown action: {action}")
                break
        
        # Safety: Max iterations reached
        print(f"⚠️ Max iterations ({max_iterations}) reached")
        
        return {
            "type": "error",
            "error": "Max iterations reached - workflow did not complete",
            "state": self._get_state_snapshot()
        }
    
    def _execute_agent(self, agent_name, context):
        """
        Execute a specific agent with appropriate inputs
        
        Args:
            agent_name: Name of agent to execute
            context: User's original message (for parsing)
        
        Returns:
            Agent result dict
        """
        
        print(f"\n⚡ Executing: {agent_name}")
        
        agent = self.agents[agent_name]
        
        # Route to correct agent with correct inputs
        
        if agent_name == "ParsingAgent":
            # Parsing needs the raw user message
            return agent.execute(context)
        
        elif agent_name == "FormAnalysisAgent":
            # Form analysis needs parsed data
            return agent.execute(self.state["collected_data"])
        
        elif agent_name == "InjuryDiagnosisAgent":
            # Diagnosis needs parsed data + form analysis
            form_result = self.state["agent_results"].get("FormAnalysisAgent", {})
            return agent.execute(self.state["collected_data"], form_result)
        
        elif agent_name == "ResearchAgent":
            # Research needs diagnosis + parsed data
            diagnosis = self.state["agent_results"].get("InjuryDiagnosisAgent", {})
            return agent.execute(diagnosis, self.state["collected_data"])
        
        elif agent_name == "PrescriptionAgent":
            # Prescription needs everything
            diagnosis = self.state["agent_results"].get("InjuryDiagnosisAgent", {})
            research = self.state["agent_results"].get("ResearchAgent", {})
            form = self.state["agent_results"].get("FormAnalysisAgent", {})
            return agent.execute(diagnosis, research, form, self.state["collected_data"])
        
        else:
            print(f"❌ Unknown agent: {agent_name}")
            return {"error": f"Unknown agent: {agent_name}"}
    
    def _compile_final_result(self):
        """
        Compile all agent results into final output
        """
        
        agent_results = self.state["agent_results"]
        
        # Extract key information
        parsed_data = self.state.get("collected_data", {})
        
        form_analysis = agent_results.get("FormAnalysisAgent", {}).get("form_analysis", "N/A")
        
        diagnosis = agent_results.get("InjuryDiagnosisAgent", {}).get("diagnosis", "N/A")
        
        research = agent_results.get("ResearchAgent", {})
        research_synthesis = research.get("synthesis", "N/A")
        web_sources = research.get("sources", [])
        web_results = research.get("web_results", [])
        
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
        """Get current state for debugging/API response"""
        return {
            "collected_data": self.state["collected_data"],
            "confidence": self.state["confidence"],
            "last_agent": self.state["last_agent"],
            "agents_executed": self.planner.execution_history,
            "conversation_turns": len(self.conversation_manager.conversation_history)
        }


# =====================================================
# 🚀 Entry Points
# =====================================================

def run_conversation(user_message):
    """
    Entry point for conversational flow
    Processes ONE message at a time
    
    Args:
        user_message: User's message
        
    Returns:
        Response dict with type and content
    """
    
    # Create or load orchestrator
    # In real implementation, this would be session-based
    orchestrator = DynamicMasterOrchestrator()
    
    # Process message
    result = orchestrator.process_user_message(user_message)
    
    return result


def run_single_shot(user_input):
    """
    Entry point for single-shot analysis (backward compatibility)
    Processes entire flow in one go
    
    Args:
        user_input: Complete user description
        
    Returns:
        Final result dict
    """
    
    buf = StringIO()
    orchestrator = DynamicMasterOrchestrator()
    
    with redirect_stdout(buf):
        # Keep processing until complete
        result = orchestrator.process_user_message(user_input)
        
        # If it asks questions, auto-skip them for single-shot mode
        while result["type"] == "question":
            # Force proceed without additional info
            orchestrator.conversation_manager.force_proceed()
            result = orchestrator.process_user_message("Continue with what you have")
        
        if result["type"] == "complete":
            final = result["result"]
            
            print("\n" + "="*70)
            print("📊 FINAL RESULTS")
            print("="*70)
            print("\n🎯 ACTION PLAN:")
            print(final["action_plan"])
    
    printed_output = buf.getvalue()
    
    return {
        "result": result.get("result", result),
        "printed": printed_output
    }


def main():
    """
    Main entry point for testing
    """
    print("🧪 Testing Dynamic Master Orchestrator...\n")
    
    # Test input
    test_input = "Sharp right knee pain during squat ascent"
    
    result = run_single_shot(test_input)
    
    print(result["printed"])
    
    return result


if __name__ == "__main__":
    main()
