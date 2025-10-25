
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_agent import BaseAgent
from tools import WebSearchTool

class ResearchAgent(BaseAgent):
    """Searches knowledge base AND web for supporting evidence"""
    
    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            role="researching evidence from knowledge base and web sources"
        )
        self.web_search_tool = WebSearchTool()
    
    def search_knowledge_base(self, query):
        """Placeholder - will use ChromaDB later"""
        return [{
            "source": "Knowledge Base",
            "content": "Placeholder: KB search coming soon",
            "relevance": "medium"
        }]
    
    def execute(self, diagnosis, exercise_info):
        """Research evidence"""
        
        exercise = exercise_info.get("exercise", "workout")
        pain_location = exercise_info.get("pain_location", "pain")
        
        kb_query = f"{exercise} {pain_location} form correction"
        web_query = f"{exercise} {pain_location} injury causes treatment"
        
        print(f"📚 KB search: {kb_query}")
        kb_results = self.search_knowledge_base(kb_query)
        
        print(f"🌐 Web search: {web_query}")
        web_results = self.web_search_tool.search(web_query, num_results=3)
        
        web_snippets = "\n".join([f"- {r['title']}: {r['snippet'][:150]}" for r in web_results])
        
        prompt = f"""Synthesize research findings:

DIAGNOSIS: {diagnosis.get('diagnosis', 'N/A')}
WEB FINDINGS: {web_snippets}

Provide:
1. KEY EVIDENCE (2-3 points citing sources)
2. CREDIBILITY (how well research supports diagnosis)

Keep under 150 words."""
        
        synthesis = self.call_llm(prompt, max_tokens=300)
        
        self.log_action("research_evidence", {
            "tool": "WebSearchTool",
            "results": len(web_results)
        })
        
        return {
            "agent": self.name,
            "kb_results": kb_results,
            "web_results": web_results,
            "synthesis": synthesis,
            "sources": [r['url'] for r in web_results if r.get('url')],
            "status": "success"
        }
