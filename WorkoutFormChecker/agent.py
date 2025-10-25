from openai import OpenAI
from dotenv import load_dotenv
import os

# ============ CONFIGURATION ============

load_dotenv('/content/drive/MyDrive/WorkoutFormChecker/.env')
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
RETRIEVER_KEY = os.getenv("RETRIEVER_KEY")

# ============ LLM HELPER ============

def call_llm(prompt, max_tokens=1024):
    """Call Nemotron LLM"""
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY
    )
    
    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-nano-8b-v1",
        messages=[{"role":"user","content":prompt}],  # ✅ USE PROMPT HERE!
        temperature=0.7,
        max_tokens=max_tokens,
        stream=False
    )
    
    return completion.choices[0].message.content
# ============ RETRIEVER FUNCTIONS ============

def get_embedding(text):
    """Get embeddings from NVIDIA API"""
    
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",  # Fixed: base_url not url, added v1
        api_key=RETRIEVER_KEY
    )
    
    response = client.embeddings.create(
        input=[text],  # Fixed: use the text parameter, not hardcoded
        model="nvidia/nv-embedqa-e5-v5",
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"}
    )  # Fixed: ) not }
    
    return response.data[0].embedding

def retrieve_documents(query, collection_name, top_k=3):
    """Retrieve relevant documents from ChromaDB"""
    # This will be implemented after we set up ChromaDB
    # For now, return empty list
    return []

print("Retriever functions added!")
# ============ DATA MODELS ============

class AgentStep:
    """Represents one step in the agent's reasoning chain"""
    
    def __init__(self, step_number, action, reasoning, retrieved_docs=[], llm_output=""):
        self.step_number = step_number
        self.action = action
        self.reasoning = reasoning
        self.retrieved_docs = retrieved_docs
        self.llm_output = llm_output
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "step_number": self.step_number,
            "action": self.action,
            "reasoning": self.reasoning,
            "retrieved_docs": self.retrieved_docs,
            "llm_output": self.llm_output
        }

print("AgentStep class added!")

# ============ AGENT STATE ============

class AgentState:
    """Tracks agent's current state and decisions"""
    
    def __init__(self):
        self.steps = []
        self.current_step = 0
        self.max_iterations = 7
        self.extracted_info = {}
        self.retrieved_context = {}
        self.root_cause = None
        self.is_done = False
    
    def add_step(self, action, reasoning, retrieved_docs=[], llm_output=""):
        """Add a step to the reasoning chain"""
        self.current_step += 1
        step = AgentStep(
            step_number=self.current_step,
            action=action,
            reasoning=reasoning,
            retrieved_docs=retrieved_docs,
            llm_output=llm_output
        )
        self.steps.append(step)
        return step
    
    def should_continue(self):
        """Check if agent should keep reasoning"""
        return not self.is_done and self.current_step < self.max_iterations

print("AgentState class added!")

# ============ AGENT ORCHESTRATOR ============

class WorkoutAgent:
    """Main agentic orchestrator - makes decisions and coordinates everything"""
    
    def __init__(self):
        self.state = None
    
    def run(self, user_input):
        """Main agentic loop"""
        # Initialize state
        self.state = AgentState()
        
        # Step 1: Parse user input
        self._parse_input(user_input)
        
        # Step 2: Agentic reasoning loop
        while self.state.should_continue():
            next_action = self._decide_next_action()
            
            if next_action == "RETRIEVE_FORM":
                self._retrieve_form_guides()
            elif next_action == "RETRIEVE_INJURY":
                self._retrieve_injury_patterns()
            elif next_action == "REASON":
                self._reason_root_cause()
            elif next_action == "RETRIEVE_CORRECTIVE":
                self._retrieve_correctives()
            elif next_action == "GENERATE":
                final_answer = self._generate_plan()
                self.state.is_done = True
                return final_answer, self.state.steps
        
        return "Could not complete analysis", self.state.steps
    
    def _parse_input(self, user_input):
        """Parse user input to extract key info"""
        prompt = f"""Extract key information from this workout issue:
        
User Input: {user_input}

Extract and respond ONLY with JSON format:
{{
    "exercise": "name of exercise",
    "pain_location": "where it hurts",
    "pain_timing": "when during movement"
}}"""
        
        response = call_llm(prompt, max_tokens=200)
        
        # Try to parse JSON
        import json
        try:
            self.state.extracted_info = json.loads(response)
        except:
            self.state.extracted_info = {"exercise": "unknown"}
        
        self.state.add_step(
            "PARSE_INPUT",
            "Extracted key information from user input",
            llm_output=response
        )
    
    def _decide_next_action(self):
        """Agent decides what to do next"""
        if "form_guides" not in self.state.retrieved_context:
            return "RETRIEVE_FORM"
        elif "injury_patterns" not in self.state.retrieved_context:
            return "RETRIEVE_INJURY"
        elif not self.state.root_cause:
            return "REASON"
        elif "correctives" not in self.state.retrieved_context:
            return "RETRIEVE_CORRECTIVE"
        else:
            return "GENERATE"
    
    def _retrieve_form_guides(self):
        """Retrieve exercise form guides"""
        exercise = self.state.extracted_info.get("exercise", "squat")
        docs = retrieve_documents(f"{exercise} proper form", "form_guides", top_k=3)
        
        self.state.retrieved_context["form_guides"] = docs
        self.state.add_step(
            "RETRIEVE_FORM",
            f"Retrieved form guides for {exercise}",
            retrieved_docs=docs
        )
    
    def _retrieve_injury_patterns(self):
        """Retrieve similar injury patterns"""
        pain = self.state.extracted_info.get("pain_location", "")
        docs = retrieve_documents(f"injury {pain}", "injury_patterns", top_k=3)
        
        self.state.retrieved_context["injury_patterns"] = docs
        self.state.add_step(
            "RETRIEVE_INJURY",
            "Retrieved similar injury patterns",
            retrieved_docs=docs
        )
    
    def _reason_root_cause(self):
        """Reason about root cause"""
        form_guides = "\n".join(self.state.retrieved_context.get("form_guides", []))
        injury_patterns = "\n".join(self.state.retrieved_context.get("injury_patterns", []))
        
        prompt = f"""Analyze this workout issue and determine ROOT CAUSE:

Exercise: {self.state.extracted_info.get('exercise')}
Pain: {self.state.extracted_info.get('pain_location')} during {self.state.extracted_info.get('pain_timing')}

Form Guidelines:
{form_guides if form_guides else "No form guides available"}

Injury Patterns:
{injury_patterns if injury_patterns else "No injury patterns available"}

Determine the ROOT CAUSE in 1-2 sentences."""
        
        response = call_llm(prompt, max_tokens=300)
        self.state.root_cause = response
        
        self.state.add_step(
            "REASON",
            "Analyzed and determined root cause",
            llm_output=response
        )
    
    def _retrieve_correctives(self):
        """Retrieve corrective exercises"""
        docs = retrieve_documents(f"corrective exercises {self.state.root_cause[:100]}", "correctives", top_k=3)
        
        self.state.retrieved_context["correctives"] = docs
        self.state.add_step(
            "RETRIEVE_CORRECTIVE",
            "Retrieved corrective exercises",
            retrieved_docs=docs
        )
    
    def _generate_plan(self):
        """Generate final action plan"""
        correctives = "\n".join(self.state.retrieved_context.get("correctives", []))
        
        prompt = f"""You are an expert strength coach. Create a CONCISE, SPECIFIC action plan.

        ROOT CAUSE: {self.state.root_cause}

        Corrective Exercises:
        {correctives if correctives else "General mobility work"}

        Format EXACTLY like this:

        🎯 ROOT CAUSE
        [1-2 sentences explaining the issue]

        ⚡ IMMEDIATE ACTION (Next Workout)
        - [Specific change 1]
        - [Specific change 2]
        - [Specific change 3]

        🔧 THIS WEEK (Daily Work)
        - [Exercise 1: 3x15 reps]
        - [Exercise 2: 10 minutes daily]
        - [Exercise 3: Before each workout]

        📊 MONITOR
        - [Track this metric]
        - [Reassess in X days]

        ⚠️ SEE A PRO IF
        - [Red flag 1]
        - [Red flag 2]

        Keep under 300 words. Be SPECIFIC, not generic. No "Additional Tips" section.
        """
        
        response = call_llm(prompt, max_tokens=800)
        
        self.state.add_step(
            "GENERATE",
            "Generated personalized action plan",
            llm_output=response
        )
        
        return response

print("WorkoutAgent class created!")


if __name__ == "__main__":
  print(" Agent module loaded!")
  agent = WorkoutAgent()
  print("Agent instantiated successfully!")