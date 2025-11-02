# planner_agent.py

import os
import logging
from base_agent import BaseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """
    Orchestrates workflow - decides which agent to run next.
    """
    AGENT_SEQUENCE = [
        "ParsingAgent",
        "FormAnalysisAgent",
        "InjuryDiagnosisAgent",
        "ResearchAgent",
        "PrescriptionAgent"
    ]

    def __init__(self):
        super().__init__(name="PlannerAgent", role="orchestrating workflow")
        self.execution_history = []

    def decide_next_action(self, state):
            """Decide what to do next"""
            
            collected_data = state.get("collected_data", {})
            conversation_turns = state.get("conversation_turns", 0)
            last_agent = state.get("last_agent", None)
            execution_history = self.execution_history
            needs_clarification = state.get("needs_clarification", False)

            logger.info("🧠 Planner deciding...")
            logger.info(f"   Agents run: {len(execution_history)} total")
            logger.info(f"   Needs clarification: {needs_clarification}")
            logger.info(f"   Collected data: {list(collected_data.keys())}")

            # ✅ RULE 1: Always run ParsingAgent first
            # ✅ Always ensure FormAnalysisAgent runs before Diagnosis
            if "ParsingAgent" in execution_history and "FormAnalysisAgent" not in execution_history:
                logger.info("   → Ensuring FormAnalysisAgent runs before any diagnosis or research")
                return {
                    "action": "run_agent",
                    "agent": "FormAnalysisAgent",
                    "reason": "Mandatory form analysis before diagnosis"
                }

            
            # ✅ RULE 2: If conversation_manager needs info, ask user
            if needs_clarification and conversation_turns < 15:
                logger.info("   → Need more information from user")
                return {
                    "action": "ask_user",
                    "question": None,  # Master will call conversation_manager
                    "reason": "Gathering required/optional info"
                }
            
            # ✅ RULE 3: Prevent same agent running twice in a row
            if execution_history and execution_history[-1] == last_agent:
                logger.warning(f"   ⚠️ {last_agent} just ran, choosing next agent")
                
                # Get next agent in sequence
                try:
                    current_idx = self.AGENT_SEQUENCE.index(last_agent)
                    if current_idx + 1 < len(self.AGENT_SEQUENCE):
                        next_agent = self.AGENT_SEQUENCE[current_idx + 1]
                        if next_agent not in execution_history:
                            logger.info(f"   → Sequential: {next_agent} after {last_agent}")
                            return {
                                "action": "run_agent",
                                "agent": next_agent,
                                "reason": f"Sequential execution after {last_agent}"
                            }
                except ValueError:
                    pass
            
            # ✅ RULE 4: Use static sequence - run next unexecuted agent
            for agent_name in self.AGENT_SEQUENCE:
                if agent_name not in execution_history:
                    logger.info(f"   → Next in sequence: {agent_name}")
                    return {
                        "action": "run_agent",
                        "agent": agent_name,
                        "reason": f"Static sequence: {agent_name}"
                    }
            
            # ✅ RULE 5: All agents done - complete
            logger.info("   → All agents executed, completing")
            return {
                "action": "complete",
                "reason": "All agents executed"
            }




    def record_agent_execution(self, agent_name):
        self.execution_history.append(agent_name)
        self.log_action("agent_executed", {"agent": agent_name})

    def get_execution_summary(self):
        return {
            "total_agents_run": len(self.execution_history),
            "agents_executed": self.execution_history,
            "unique_agents": list(set(self.execution_history))
        }