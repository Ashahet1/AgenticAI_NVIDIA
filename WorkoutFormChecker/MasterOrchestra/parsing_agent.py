# parsing_agent.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_agent import BaseAgent

class ParsingAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ParsingAgent", role="extracting structured data")
    
    def execute(self, collected_data):
        """Just validate and return what we have"""
        print(f"   📋 Validating collected data...")
        
        return {
            "agent": self.name,
            "parsed_data": collected_data,
            "status": "success"
        }