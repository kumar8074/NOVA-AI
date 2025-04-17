import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GEMINI_API_KEY=os.getenv("GOOGLE_API_KEY")

# Model settings
LLM_MODEL = "Deepseek-R1-Distill-Qwen-32b"


# Get current date
CURRENT_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# Document processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
