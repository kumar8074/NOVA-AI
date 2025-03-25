import os
import datetime
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model settings
LLM_MODEL = "Deepseek-R1-Distill-Qwen-32b"
try:
    EMBEDDING_MODEL = CohereEmbeddings(model="embed-english-v3.0") 
except Exception:
    EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Get current date
CURRENT_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# Document processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200