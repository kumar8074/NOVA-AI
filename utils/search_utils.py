from tavily import TavilyClient
from config import TAVILY_API_KEY

# Initialize Tavily client
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def perform_web_search(query: str) -> str:
    """Perform a web search using the Tavily API"""
    try:
        response = tavily_client.search(query, search_depth="advanced", max_results=5)
        if response and "results" in response and len(response["results"]) > 0:
            formatted_results = []
            for i, res in enumerate(response["results"], 1):
                title = res.get('title', 'No title')
                url = res.get('url', '#')
                content = res.get('content', 'No description available.')
                
                # Format the result with source number for easier reference
                formatted_results.append(f"Source {i}: {title}\nURL: {url}\nContent: {content}\n")
            
            return "\n".join(formatted_results)
        return "No relevant search results found."
    except Exception as e:
        return f"Error performing web search: {str(e)}"

def needs_web_search(query):
    """Determine if a web search is needed based on the query"""
    # Check for explicit request for search
    explicit_search_patterns = [
        "search for", "look up", "find information", "search the web",
        "what's the latest", "current news", "recent updates"
    ]
    
    query_lower = query.lower()
    
    for pattern in explicit_search_patterns:
        if pattern in query_lower:
            return True
    
    # Check for queries about current events, dates, or time-sensitive information
    time_sensitive_patterns = [
        "today", "current", "latest", "recent", "now", "update", 
        "news", "weather", "price", "stock", "bitcoin", "crypto",
        "happened", "trending", "score", "result", "happening"
    ]
    
    for pattern in time_sensitive_patterns:
        if pattern in query_lower:
            return True
    
    # Check for questions about specific factual information that might need verification
    factual_patterns = [
        "how many", "how much", "what is the population", "what is the distance",
        "how far", "how old", "when was", "where is", "who is the current"
    ]
    
    for pattern in factual_patterns:
        if pattern in query_lower:
            return True
    
    # Don't use web search for conversational queries
    conversational_patterns = [
        "how are you", "what do you think", "can you help",
        "who are you", "tell me about yourself"
    ]
    
    for pattern in conversational_patterns:
        if pattern in query_lower:
            return False
    
    # Default to not using web search for most queries
    return False