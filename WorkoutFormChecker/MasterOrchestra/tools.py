import requests
import os
import fitz
import json
import numpy as np
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv('/content/drive/MyDrive/WorkoutFormChecker/.env')

class WebSearchTool:
    """
    Web Search Tool - Uses Brave Search API for REAL web search
    """
    def __init__(self):
        self.name = "WebSearchTool"
        self.api_key = os.getenv("BRAVE_API_KEY", "")
        
    def search(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        """
        Search the web and return REAL results from Brave Search
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of dicts with title, snippet, url
        """
        if not self.api_key:
            logger.error("No Brave API key found!")
            return [{
                'title': 'Configuration Error',
                'snippet': 'Brave API key not configured. Web search unavailable.',
                'url': ''
            }]
        
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key
            }
            params = {
                "q": query,
                "count": num_results
            }
            
            logger.info(f"Calling Brave Search API for: '{query}'")
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                web_results = data.get('web', {}).get('results', [])
                for item in web_results[:num_results]:
                    results.append({
                        'title': item.get('title', 'No title'),
                        'snippet': item.get('description', 'No description'),
                        'url': item.get('url', '')
                    })
                if results:
                    logger.info(f"Found {len(results)} real results from Brave Search")
                    return results
                else:
                    logger.warning("No results found for this query")
                    return [{
                        'title': 'No Results Found',
                        'snippet': f'No web results found for: {query}. Try different search terms.',
                        'url': ''
                    }]
            elif response.status_code == 401:
                logger.error("Invalid Brave API key")
                return [{
                    'title': 'Authentication Error',
                    'snippet': 'Invalid or expired Brave API key.',
                    'url': ''
                }]
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded")
                return [{
                    'title': 'Rate Limit Exceeded',
                    'snippet': 'Too many requests. Please try again later.',
                    'url': ''
                }]
            else:
                logger.error(f"Brave API error: {response.status_code}")
                return [{
                    'title': 'API Error',
                    'snippet': f'Web search failed with status code: {response.status_code}',
                    'url': ''
                }]
        except requests.exceptions.Timeout:
            logger.warning("Web search timed out")
            return [{
                'title': 'Search Timeout',
                'snippet': 'Web search request timed out. Please try again.',
                'url': ''
            }]
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return [{
                'title': 'Search Error',
                'snippet': f'An error occurred during web search: {str(e)}',
                'url': ''
            }]

logger.info("WebSearchTool created (REAL search only, no fake fallback)!")

class KnowledgeBaseTool:
    """
    Uses nv-embedqa-e5-v5 model from Bedrock to embed and retrieve from PDFs in knowledge_base/docs
    """
    def __init__(self, kb_path: str = "knowledge_base/docs", kb_dir: Optional[str] = None):
        # Accept either kb_path or kb_dir (kb_dir kept for compatibility)
        if kb_dir is not None:
            kb_path = kb_dir

        logger.info("Initializing Knowledge Base Tool with nv-embedqa-e5-v5 ...")
        self.kb_path = kb_path

        self.api_key = os.environ.get("RETRIEVER_KEY")
        self.model = "nv-embedqa-e5-v5"
        self.model_url = "https://integrate.api.nvidia.com/v1/embeddings"

        parent_dir = os.path.dirname(self.kb_path) or "."
        os.makedirs(parent_dir, exist_ok=True)
        self.index_file = os.path.join(parent_dir, "kb_index.json")

        self.index = []
        self.embeddings = []
        self._load_or_build_index()

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        doc = fitz.open(pdf_path)
        return "\n".join([page.get_text("text") for page in doc])

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using NVIDIA nv-embedqa-e5-v5"""
        if not texts or not isinstance(texts, list):
            logger.warning("No valid texts provided for embedding.")
            return np.zeros((0, 1024), dtype=np.float32)

        headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json"
        }

        body = {
            "model": self.model,
            "input": texts,
            "input_type": "document",
            "encoding_format": "float",
            "truncate": "NONE"
        }

        try:
            response = requests.post(
                f"{self.model_url}/embeddings",
                headers=headers,
                json=body,
                timeout=60
            )

            if response.status_code != 200:
                logger.error(f"Embedding API Error: {response.status_code} - {response.text}")
                return np.zeros((len(texts), 1024), dtype=np.float32)

            data = response.json()
            embeddings = [item["embedding"] for item in data.get("data", [])]
            if len(embeddings) != len(texts):
                logger.warning("Embedding response size mismatch; padding/trimming embeddings.")
            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            logger.error(f"Exception during embedding: {e}")
            return np.zeros((len(texts), 1024), dtype=np.float32)

    def _load_or_build_index(self) -> List[Dict[str, Any]]:
        """Either load cached embeddings or build from scratch"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r") as f:
                    self.index = json.load(f)
                    logger.info(f"Loaded KB index from {self.index_file} ({len(self.index)} chunks).")
                    return self.index
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e} - rebuilding.")

        logger.info(f"Building new KB index from PDFs in {self.kb_path} ...")
        index = []
        if not os.path.exists(self.kb_path):
            logger.warning(f"KB path {self.kb_path} does not exist. No PDFs to index.")
            self.index = []
            return self.index

        for fname in os.listdir(self.kb_path):
            if not fname.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(self.kb_path, fname)
            try:
                text = self._extract_text_from_pdf(pdf_path)
            except Exception as e:
                logger.warning(f"Failed to extract text from {pdf_path}: {e}")
                continue

            chunks = [text[i:i+800] for i in range(0, len(text), 800) if text[i:i+800].strip()]
            if not chunks:
                continue

            embeddings = self._embed_texts(chunks)
            if embeddings.shape[0] != len(chunks):
                embeddings = np.zeros((len(chunks), 1024), dtype=np.float32)

            for chunk, emb in zip(chunks, embeddings):
                index.append({
                    "doc": fname,
                    "content": chunk,
                    "embedding": emb.tolist()
                })

        try:
            with open(self.index_file, "w") as f:
                json.dump(index, f)
            logger.info(f"KB index built with {len(index)} chunks and saved to {self.index_file}.")
        except Exception as e:
            logger.warning(f"Failed to save KB index: {e}")

        self.index = index
        return index

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find top relevant chunks for a given query"""
        q_embs = self._embed_texts([query])
        if q_embs is None or q_embs.shape[0] == 0:
            logger.warning("Query embedding failed; returning empty results.")
            return []

        q_emb = q_embs[0]
        scored = []
        for item in self.index:
            emb = np.array(item["embedding"], dtype=np.float32)
            if np.linalg.norm(q_emb) == 0 or np.linalg.norm(emb) == 0:
                score = 0.0
            else:
                score = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [
            {
                "title": item["doc"],
                "content": item["content"][:400],
                "score": float(score),
                "source": f"{self.kb_path}/{item['doc']}"
            }
            for score, item in scored[:n_results]
        ]
        return results