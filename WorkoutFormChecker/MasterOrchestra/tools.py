
import requests
import os
from dotenv import load_dotenv

load_dotenv('/content/drive/MyDrive/WorkoutFormChecker/.env')

class WebSearchTool:
    """
    Web Search Tool - Uses Brave Search API for REAL web search
    """
    
    def __init__(self):
        self.name = "WebSearchTool"
        self.api_key = os.getenv("BRAVE_API_KEY", "")
        
    def search(self, query, num_results=3):
        """
        Search the web and return REAL results from Brave Search
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of dicts with title, snippet, url
        """
        if not self.api_key:
            print("❌ ERROR: No Brave API key found!")
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
            
            print(f"🔍 Calling Brave Search API for: '{query}'")
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Extract web results
                web_results = data.get('web', {}).get('results', [])
                
                for item in web_results[:num_results]:
                    results.append({
                        'title': item.get('title', 'No title'),
                        'snippet': item.get('description', 'No description'),
                        'url': item.get('url', '')
                    })
                
                if results:
                    print(f"✅ Found {len(results)} real results from Brave Search")
                    return results
                else:
                    print("⚠️ No results found for this query")
                    return [{
                        'title': 'No Results Found',
                        'snippet': f'No web results found for: {query}. Try different search terms.',
                        'url': ''
                    }]
                    
            elif response.status_code == 401:
                print("❌ ERROR: Invalid Brave API key")
                return [{
                    'title': 'Authentication Error',
                    'snippet': 'Invalid or expired Brave API key.',
                    'url': ''
                }]
            elif response.status_code == 429:
                print("⚠️ Rate limit exceeded")
                return [{
                    'title': 'Rate Limit Exceeded',
                    'snippet': 'Too many requests. Please try again later.',
                    'url': ''
                }]
            else:
                print(f"❌ Brave API error: {response.status_code}")
                return [{
                    'title': 'API Error',
                    'snippet': f'Web search failed with status code: {response.status_code}',
                    'url': ''
                }]
                
        except requests.exceptions.Timeout:
            print("⚠️ Web search timed out")
            return [{
                'title': 'Search Timeout',
                'snippet': 'Web search request timed out. Please try again.',
                'url': ''
            }]
        except Exception as e:
            print(f"❌ Web search error: {str(e)}")
            return [{
                'title': 'Search Error',
                'snippet': f'An error occurred during web search: {str(e)}',
                'url': ''
            }]

print("✅ WebSearchTool created (REAL search only, no fake fallback)!")
