# research_agent.py
from openai import OpenAI
import requests
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_agent import BaseAgent
from tools import WebSearchTool
import fitz  # PyMuPDF
import json
import boto3
import numpy as np

class KnowledgeBaseTool:
    """
    Uses nv-embedqa-e5-v5 model from Bedrock to embed and retrieve from PDFs in knowledge_base/docs
    """
    def __init__(self, kb_path="knowledge_base/docs"):
      print("🔧 Initializing Knowledge Base Tool with nv-embedqa-e5-v5 ...")
      self.kb_path = kb_path
      self.client = OpenAI(
          api_key=os.environ.get("NVIDIA_API_KEY"),
          base_url="https://integrate.api.nvidia.com/v1"
        )
      self.index = []
      self.embeddings = []
      self._build_kb_index()

    def _extract_text_from_pdf(self, pdf_path):
      """Extract text from a PDF file"""
      doc = fitz.open(pdf_path)
      return "\n".join([page.get_text("text") for page in doc])

    def _embed_texts(self, texts):
        """Embed texts using NVIDIA nv-embedqa-e5-v5"""
        if not texts or not isinstance(texts, list):
            print("⚠️ No valid texts provided for embedding.")
            return [np.zeros(1024)] * len(texts)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        body = {
            "model": self.model,
            "input": texts,
            "input_type": "document",  # ✅ required for asymmetric QA models
            "encoding_format": "float",
            "truncate": "NONE"
        }

        try:
            response = requests.post(
                f"{self.model_url}/embeddings",  # ✅ corrected full endpoint
                headers=headers,
                json=body,
                timeout=60
            )

            if response.status_code != 200:
                print(f"❌ Embedding API Error: {response.status_code} - {response.text}")
                return [np.zeros(1024)] * len(texts)

            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            print(f"❌ Exception during embedding: {e}")
            return [np.zeros(1024)] * len(texts)

    def _load_or_build_index(self):
        """Either load cached embeddings or build from scratch"""
        if os.path.exists(self.index_file):
            with open(self.index_file, "r") as f:
                return json.load(f)

        print(f"⚙️ Building new KB index from PDFs in {self.kb_dir} ...")
        index = []
        for fname in os.listdir(self.kb_path):
            if not fname.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(self.kb_path, fname)
            text = self._extract_text_from_pdf(pdf_path)
            chunks = [text[i:i+800] for i in range(0, len(text), 800)]
            embeddings = self._embed_texts(chunks)
            for chunk, emb in zip(chunks, embeddings):
                index.append({
                    "doc": fname,
                    "content": chunk,
                    "embedding": emb.tolist()
                })

        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        with open(self.index_file, "w") as f:
            json.dump(index, f)

        print(f"✅ KB index built with {len(index)} chunks.")
        return index

    def search(self, query, n_results=5):
        """Find top relevant chunks for a given query"""
        q_emb = self._embed_texts([query])[0]
        scored = []
        for item in self.index:
            emb = np.array(item["embedding"])
            score = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [
            {
                "title": item["doc"],
                "content": item["content"][:400],
                "score": float(score),
                "source": f"knowledge_base/docs/{item['doc']}"
            }
            for score, item in scored[:n_results]
        ]
        return results

class ResearchAgent(BaseAgent):
    """
    Researches evidence from knowledge base AND/OR web
    NOW with intelligent tool selection based on query type
    """
    
    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            role="researching evidence and selecting appropriate tools"
        )
        self.web_search_tool = WebSearchTool()
        
        # Initialize the Knowledge Base tool (using NVIDIA nv-embedqa-e5-v5)
        try:
            print("🔧 Initializing Knowledge Base Tool with nv-embedqa-e5-v5 ...")
            self.kb_tool = KnowledgeBaseTool(kb_dir="knowledge_base/docs")
            print("✅ Knowledge Base Tool loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to initialize KnowledgeBaseTool: {e}")
            self.kb_tool = None
    
    def decide_tool(self, diagnosis_data, exercise_info):
        """
        🧠 INTELLIGENT TOOL DECISION
        
        Analyzes the case and decides which tool to use:
        - Knowledge Base: Common injuries with established protocols
        - Web Search: Rare injuries, latest research, controversial topics
        - Both: Complex or uncertain cases
        
        Args:
            diagnosis_data: Results from InjuryDiagnosisAgent
            exercise_info: Parsed exercise information
        
        Returns:
            str: "knowledge_base" | "web_search" | "both"
        """
        
        diagnosis_text = diagnosis_data.get("diagnosis", "").lower()
        exercise = exercise_info.get("exercise", "").lower()
        confidence = diagnosis_data.get("confidence", "unknown")
        
        print(f"\n🧠 ResearchAgent deciding tool...")
        print(f"   - Exercise: {exercise}")
        print(f"   - Diagnosis confidence: {confidence}")
        print(f"   - Diagnosis snippet: {diagnosis_text[:100]}...")
        
        # DECISION LOGIC
        
        # Common injury patterns that we likely have KB data for
        common_injury_patterns = [
            "knee pain",
            "patellar tendinitis",
            "patellar tendonitis",
            "it band",
            "runner's knee",
            "jumper's knee",
            "shoulder impingement",
            "rotator cuff",
            "lower back pain",
            "lumbar strain",
            "hip flexor",
            "hamstring strain",
            "tennis elbow",
            "golfer's elbow",
            "shin splints",
            "plantar fasciitis",
            "achilles tendinitis",
            "groin strain",
            "quad strain",
            "calf strain"
        ]
        
        # Check if diagnosis mentions any common patterns
        is_common_injury = any(pattern in diagnosis_text for pattern in common_injury_patterns)
        
        # Common exercises with well-documented form issues
        common_exercises = [
            "squat", "bench press", "deadlift", "overhead press",
            "pull up", "row", "lunge", "leg press"
        ]
        is_common_exercise = any(ex in exercise for ex in common_exercises)
        
        # DECISION TREE
        
        # Case 1: Common injury + Common exercise + Medium/High confidence
        # → Use Knowledge Base (we likely have good protocols)
        if is_common_injury and is_common_exercise and confidence in ["medium", "high"]:
            decision = "knowledge_base"
            reason = f"Common injury pattern ({diagnosis_text[:50]}...) with established protocols - using Knowledge Base"
        
        # Case 2: Low confidence OR uncommon pattern
        # → Use Web Search (need latest research and multiple perspectives)
        elif confidence == "low" or not is_common_injury:
            decision = "web_search"
            reason = f"{'Low confidence diagnosis' if confidence == 'low' else 'Uncommon injury pattern'} - using Web Search for latest research"
        
        # Case 3: Common injury but want to validate with latest research
        # → Use BOTH (KB for protocols + Web for validation)
        elif is_common_injury and confidence == "medium":
            decision = "both"
            reason = "Common injury but medium confidence - using both KB (protocols) and Web (validation)"
        
        # Case 4: Fallback - use Web Search if KB not available yet
        else:
            if self.kb_tool is None:
                decision = "web_search"
                reason = "Knowledge Base not yet available - using Web Search"
            else:
                decision = "knowledge_base"
                reason = "Using Knowledge Base for established injury protocols"
        
        print(f"   ✅ Decision: {decision.upper()}")
        print(f"   📝 Reason: {reason}")
        
        return {
            "tool": decision,
            "reason": reason,
            "is_common_injury": is_common_injury,
            "is_common_exercise": is_common_exercise
        }
    
    def execute(self, diagnosis, exercise_info):
        """
        Main execution - decides tool and executes research
        """
        
        # STEP 1: Decide which tool to use
        tool_decision = self.decide_tool(diagnosis, exercise_info)
        chosen_tool = tool_decision["tool"]
        
        # STEP 2: Execute chosen tool(s)
        if chosen_tool == "knowledge_base":
            return self._execute_kb_search(diagnosis, exercise_info, tool_decision)
        
        elif chosen_tool == "web_search":
            return self._execute_web_search(diagnosis, exercise_info, tool_decision)
        
        elif chosen_tool == "both":
            return self._execute_both_tools(diagnosis, exercise_info, tool_decision)
        
        else:
            # Fallback
            return self._execute_web_search(diagnosis, exercise_info, tool_decision)
    
    def _execute_kb_search(self, diagnosis, exercise_info, tool_decision):
        """
        Execute Knowledge Base search
        """
        exercise = exercise_info.get("exercise", "workout")
        pain_location = exercise_info.get("pain_location", "pain")
        
        kb_query = f"{exercise} {pain_location} injury treatment protocol"
        
        print(f"📚 Searching Knowledge Base: {kb_query}")
        
        # CHECK: Is KB available?
        if self.kb_tool is None:
            print("⚠️  Knowledge Base not available yet - falling back to web search")
            return self._execute_web_search(diagnosis, exercise_info, tool_decision)
        
        # Search KB
        kb_results = self.kb_tool.search(kb_query, n_results=5)
        
        # Synthesize KB findings
        synthesis = self._synthesize_kb_results(diagnosis, kb_results)
        
        self.log_action("research_evidence", {
            "tool": "knowledge_base",
            "results": len(kb_results)
        })
        
        return {
            "agent": self.name,
            "tool_used": "knowledge_base",
            "tool_decision": tool_decision,
            "kb_results": kb_results,
            "web_results": [],
            "synthesis": synthesis,
            "sources": [r.get("source", "KB") for r in kb_results],
            "status": "success"
        }
    
    def _execute_web_search(self, diagnosis, exercise_info, tool_decision):
        """
        Execute Web Search (your existing implementation)
        """
        exercise = exercise_info.get("exercise", "workout")
        pain_location = exercise_info.get("pain_location", "pain")
        
        web_query = f"{exercise} {pain_location} injury causes treatment"
        
        print(f"🌐 Searching Web: {web_query}")
        
        # Web search
        web_results = self.web_search_tool.search(web_query, num_results=3)
        
        # Create synthesis prompt
        web_snippets = "\n".join([f"- {r['title']}: {r['snippet'][:150]}" for r in web_results])
        
        prompt = f"""
You are a medical research synthesis assistant specializing in injury after excercising and exercise science literature, consider yourself as a physical Therapist.

TASK:
Integrate evidence from credible sources to support or challenge the following diagnosis.

DIAGNOSIS SUMMARY:
{diagnosis.get('diagnosis', 'N/A')}

RELEVANT WEB FINDINGS:
{web_snippets}

---

Produce a concise, evidence-based synthesis containing:

1. Key Supporting Evidence: Summarize  4-5 relevant findings or mechanisms from the web sources or literature that align with the diagnosis (e.g., overuse, improper form, muscle imbalance). Include source names or URLs if available.
2. Contradictions or Gaps (if any): Briefly mention if the web findings suggest alternate causes or lack strong consensus.
3. Credibility Assessment: Rate as Strong / Moderate / Limited and justify briefly (e.g., number of sources, study type, clinical consistency).

Write in a professional tone (as if for a clinician's research summary). Limit to 150–180 words.

IMPORTANT: Do not use any markdown formatting such as ** for bold or * for italics. Use plain text only with clear section labels.
"""
        
        synthesis = self.call_llm(prompt, max_tokens=300)
        synthesis = synthesis.replace('**', '')
        
        self.log_action("research_evidence", {
            "tool": "web_search",
            "results": len(web_results)
        })
        
        return {
            "agent": self.name,
            "tool_used": "web_search",
            "tool_decision": tool_decision,
            "kb_results": [],
            "web_results": web_results,
            "synthesis": synthesis,
            "sources": [r['url'] for r in web_results if r.get('url')],
            "status": "success"
        }
    
    def _execute_both_tools(self, diagnosis, exercise_info, tool_decision):
        """
        Execute BOTH Knowledge Base and Web Search
        """
        print(f"🔍 Using BOTH tools for comprehensive research...")
        
        # Get KB results
        kb_result = self._execute_kb_search(diagnosis, exercise_info, tool_decision)
        
        # Get Web results
        web_result = self._execute_web_search(diagnosis, exercise_info, tool_decision)
        
        # Combine and synthesize
        combined_synthesis = self._synthesize_both_results(
            diagnosis, 
            kb_result.get("kb_results", []),
            web_result.get("web_results", [])
        )
        
        self.log_action("research_evidence", {
            "tool": "both",
            "kb_results": len(kb_result.get("kb_results", [])),
            "web_results": len(web_result.get("web_results", []))
        })
        
        return {
            "agent": self.name,
            "tool_used": "both",
            "tool_decision": tool_decision,
            "kb_results": kb_result.get("kb_results", []),
            "web_results": web_result.get("web_results", []),
            "synthesis": combined_synthesis,
            "sources": kb_result.get("sources", []) + web_result.get("sources", []),
            "status": "success"
        }
    
    def _synthesize_kb_results(self, diagnosis, kb_results):
        """
        Synthesize findings from Knowledge Base
        """
        if not kb_results or len(kb_results) == 0:
            return "No relevant protocols found in Knowledge Base."
        
        # Extract KB content
        kb_content = "\n".join([
            f"- {r.get('title', 'Document')}: {r.get('content', '')[:200]}"
            for r in kb_results[:3]
        ])
        
        prompt = f"""
Based on the following protocols from our knowledge base, provide evidence-based guidance.

DIAGNOSIS:
{diagnosis.get('diagnosis', 'N/A')}

KNOWLEDGE BASE PROTOCOLS:
{kb_content}

Provide:
1. Protocol Match: How well do the KB protocols match this diagnosis?
2. Key Recommendations: What are the main treatment/rehab recommendations?
3. Evidence Quality: Rate the protocol reliability (Strong/Moderate/Limited)

Keep under 150 words. Use plain text, no markdown formatting.
"""
        
        synthesis = self.call_llm(prompt, max_tokens=250)
        return synthesis.replace('**', '')
    
    def _synthesize_both_results(self, diagnosis, kb_results, web_results):
        """
        Synthesize when using both KB and Web Search
        """
        kb_content = "\n".join([f"- KB: {r.get('content', '')[:100]}" for r in kb_results[:2]])
        web_content = "\n".join([f"- Web: {r['snippet'][:100]}" for r in web_results[:2]])
        
        prompt = f"""
Synthesize evidence from both internal protocols and latest web research.

DIAGNOSIS:
{diagnosis.get('diagnosis', 'N/A')}

INTERNAL PROTOCOLS (Knowledge Base):
{kb_content}

LATEST RESEARCH (Web):
{web_content}

Provide:
1. Agreement: Where KB protocols and web research align
2. New Insights: What recent research adds to established protocols
3. Recommendation: Best approach combining both sources

Keep under 180 words. Plain text only.
"""
        
        synthesis = self.call_llm(prompt, max_tokens=300)
        return synthesis.replace('**', '')


# Test ResearchAgent
if __name__ == "__main__":
    print("Testing ResearchAgent with tool decision logic...\n")
    
    agent = ResearchAgent()
    
    # Test 1: Common injury (should choose KB)
    diagnosis1 = {
        "diagnosis": "Likely patellar tendinitis due to overuse and quad dominance",
        "confidence": "high"
    }
    exercise1 = {
        "exercise": "squat",
        "pain_location": "right knee"
    }
    
    print("=== Test 1: Common Injury ===")
    decision1 = agent.decide_tool(diagnosis1, exercise1)
    print(f"Decision: {decision1['tool']}")
    print(f"Reason: {decision1['reason']}\n")
    
    # Test 2: Rare injury (should choose Web)
    diagnosis2 = {
        "diagnosis": "Possible scapular winging with nerve impingement",
        "confidence": "low"
    }
    exercise2 = {
        "exercise": "overhead press",
        "pain_location": "shoulder blade"
    }
    
    print("=== Test 2: Rare/Low Confidence ===")
    decision2 = agent.decide_tool(diagnosis2, exercise2)
    print(f"Decision: {decision2['tool']}")
    print(f"Reason: {decision2['reason']}\n")
    
    # Test 3: Medium confidence common injury (should choose BOTH)
    diagnosis3 = {
        "diagnosis": "Rotator cuff impingement or strain",
        "confidence": "medium"
    }
    exercise3 = {
        "exercise": "bench press",
        "pain_location": "shoulder"
    }
    
    print("=== Test 3: Medium Confidence ===")
    decision3 = agent.decide_tool(diagnosis3, exercise3)
    print(f"Decision: {decision3['tool']}")
    print(f"Reason: {decision3['reason']}\n")
    
    print("✅ ResearchAgent tests complete!")
