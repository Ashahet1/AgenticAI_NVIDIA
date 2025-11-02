
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment
#load_dotenv('/content/drive/MyDrive/WorkoutFormChecker/.env')
#NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

class BaseAgent:
    """Base class for all specialized agents"""
    
    def __init__(self, name, role):
        self.name = name
        self.role = role
        # Try EKS endpoint first
        eks_url = os.getenv("EKS_ENDPOINT", "http://a5f50a6ad507048228ecccaa20c057cd-1233535963.us-east-1.elb.amazonaws.com:8000/v1") # Delete after the demo
        eks_api_key = "not-needed"  # Local EKS NIM doesn’t need auth

        # Fallback to NVIDIA NIM endpoint if EKS unavailable
        nvidia_url = "https://integrate.api.nvidia.com/v1"
        nvidia_api_key = os.getenv("NVIDIA_API_KEY") # Need a Public API Key

        try:
            self.client = OpenAI(base_url=eks_url, api_key=eks_api_key)
            # Simple test call to check connectivity
            _ = self.client.models.list()
            print("✅ Connected to EKS endpoint.")
        except Exception as e:
            print(f"⚠️ EKS unavailable ({e}), falling back to NVIDIA endpoint.")
            if not nvidia_api_key:
                raise ValueError("NVIDIA_API_KEY not set in environment.")
            self.client = OpenAI(base_url=nvidia_url, api_key=nvidia_api_key)
            print("✅ Connected to NVIDIA endpoint.")
        
    )
        self.memory = []  # Track what this agent has done
    
    def call_llm(self, prompt, max_tokens=1024):
        """Call LLM with agent's context"""
        system_prompt = f"You are {self.name}, specialized in {self.role}. Be concise and specific."
        thinking = "on"
        try:
          messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt}
          ]
        
          response = self.client.chat.completions.create(
              model="nvidia/llama-3.1-nemotron-nano-8b-v1",
              messages= messages,
              temperature=0.7,
              max_tokens=max_tokens,
              stream=False
          )
        
          return response.choices[0].message.content
        except Exception as e:
          print(f"LLM call failes for {self.name}: {e}")
    
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
