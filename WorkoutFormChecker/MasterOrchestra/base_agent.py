
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment
load_dotenv('/content/drive/MyDrive/WorkoutFormChecker/.env')
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

class BaseAgent:
    """Base class for all specialized agents"""
    
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
        self.memory = []  # Track what this agent has done
    
    def call_llm(self, prompt, max_tokens=1024):
        """Call LLM with agent's context"""
        system_prompt = f"You are {self.name}, specialized in {self.role}. Be concise and specific."
        
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = self.client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-nano-8b-v1",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7,
            max_tokens=max_tokens,
            stream=False
        )
        return response.choices[0].message.content
    
    def log_action(self, action, result):
        """Log what this agent did"""
        self.memory.append({
            "action": action,
            "result": result,
            "agent": self.name
        })
        return result
    
    def execute(self, input_data):
        """Override this in each specialized agent"""
        raise NotImplementedError(f"{self.name} must implement execute()")

print("BaseAgent class with log_action created!")