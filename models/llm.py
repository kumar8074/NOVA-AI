from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from config import GROQ_API_KEY, LLM_MODEL, EMBEDDING_MODEL
from utils.document_utils import query_documents
from utils.search_utils import perform_web_search

def initialize_llm():
    """Initialize the LLM engine"""
    return ChatGroq(model=LLM_MODEL, groq_api_key=GROQ_API_KEY)

def initialize_embeddings():
    """Initialize the embedding model"""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

def initialize_vector_store(embedding_model):
    """Initialize the vector store"""
    return InMemoryVectorStore(embedding=embedding_model)

def get_tools():
    """Get the available tools for the assistant"""
    # Define Web Search and Document Query as Tools
    web_search_tool = Tool(
        name="Web Search",
        func=perform_web_search,
        description="Use this tool to fetch the latest information from the web. Input a search query."
    )

    document_query_tool = Tool(
        name="Document Query",
        func=query_documents,
        description="Use this tool to search through uploaded documents. Input a search query."
    )
    
    return [web_search_tool, document_query_tool]

def build_prompt_chain(message_log, system_template):
    """Build the prompt chain from message history"""
    # Start with just the system message
    messages = [SystemMessage(content=system_template)]
    
    # Add the conversation history
    for msg in message_log:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
    
    return messages