import os
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from config import GROQ_API_KEY, LLM_MODEL, COHERE_API_KEY, HUGGINGFACE_API_KEY, GEMINI_API_KEY
from utils.document_utils import query_documents
from utils.search_utils import perform_web_search

os.environ["COHERE_API_KEY"] = COHERE_API_KEY
os.environ["HUGGING_FACE_API_KEY"] = HUGGINGFACE_API_KEY
os.environ["GEMINI_API_KEY"]=GEMINI_API_KEY

def initialize_llm():
    """Initialize the LLM engine"""
    return ChatGroq(model=LLM_MODEL, groq_api_key=GROQ_API_KEY)

def initialize_embeddings():
    """Initialize the embedding model"""
    try:
        return CohereEmbeddings(model="embed-english-v3.0") 
    except Exception:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
