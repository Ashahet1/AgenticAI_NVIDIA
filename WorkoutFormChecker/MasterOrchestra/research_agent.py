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
        
        prompt = f"""
You are a medical research synthesis assistant specializing in sports injury and exercise science literature.

TASK:
Integrate evidence from credible sources to support or challenge the following diagnosis.

DIAGNOSIS SUMMARY:
{diagnosis.get('diagnosis', 'N/A')}

RELEVANT WEB FINDINGS:
{web_snippets}

---

Produce a concise, evidence-based synthesis containing:

1. Key Supporting Evidence: Summarize 2–3 relevant findings or mechanisms from the web sources or literature that align with the diagnosis (e.g., overuse, improper form, muscle imbalance). Include source names or URLs if available.
2. Contradictions or Gaps (if any): Briefly mention if the web findings suggest alternate causes or lack strong consensus.
3. Credibility Assessment: Rate as Strong / Moderate / Limited and justify briefly (e.g., number of sources, study type, clinical consistency).

Write in a professional tone (as if for a clinician's research summary). Limit to 150–180 words.

IMPORTANT: Do not use any markdown formatting such as ** for bold or * for italics. Use plain text only with clear section labels.
"""

        
        synthesis = self.call_llm(prompt, max_tokens=300)
        
        # Remove any remaining ** markdown formatting as a safety measure
        synthesis = synthesis.replace('**', '')
        
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
