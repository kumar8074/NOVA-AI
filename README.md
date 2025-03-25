# 🌌 NOVA - Intelligent Multi-Document AI Assistant

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-nova-app.streamlit.app/)
![LLM Powered](https://img.shields.io/badge/LLM-Powered-blueviolet)
![Multi-Document](https://img.shields.io/badge/Multi-Document-Analysis-success)
![Real-Time Search](https://img.shields.io/badge/Real--Time-Web%20Search-important)

**NOVA** is an advanced AI assistant with built-in multi-document intelligence and real-time web search capabilities. Designed for professionals, researchers, and curious minds, NOVA combines state-of-the-art language models with powerful document analysis tools.

![NOVA Interface Screenshot](https://via.placeholder.com/800x450.png?text=NOVA+AI+Assistant+Interface)

**Access the Deployed version of NOVA Here**: https://nova-ai-v1.streamlit.app

## ✨ Features

- **Multi-Document Intelligence**
  - 📄 Support for 15+ file types (PDF, DOCX, XLSX, PPT, CSV, JSON, etc.)
  - 🔍 Semantic search across uploaded documents
  - 📊 Structured data analysis (Excel, CSV, JSON)
- **Smart Web Integration**
  - 🌐 Real-time web search powered by Tavily
  - ⚡ Automatic search triggering based on query context
- **Advanced NLP Capabilities**
  - 💬 Natural conversation interface
  - 🤖 Groq-powered ultra-fast LLM responses
  - 🔧 Customizable system prompts and templates
- **Enterprise-Ready Features**
  - 🧠 In-memory vector store for document embeddings
  - ⚙️ Chunk-based document processing
  - 🎨 Streamlit-based professional UI

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- API keys for:
  - [Groq](https://console.groq.com/)
  - [Tavily](https://tavily.com/)
  - [Cohere](https://dashboard.cohere.com/) 
  - [HuggingFace](https://huggingface.co/settings/tokens) 

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/nova-ai-assistant.git
cd nova-ai-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys to .env

# ⚙️ Configuration
Edit the .env file with your API credentials:
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
COHERE_API_KEY=your_cohere_key  
HUGGINGFACE_API_KEY=your_hf_key 

# 💻 Usage
streamlit run app.py
```
### Basic Workflow:

1. Upload documents through the web interface

2. Ask questions about content or request analysis

3. NOVA automatically determines when to use:

- Document content

- Web search results

- Internal knowledge base

## Example Queries:

- "Summarize the key points from my PDF"

- "Compare Q2 sales figures from the Excel sheet"

- "What's the latest news about AI advancements?"

- "Create a bullet list from Section 3 of the contract"


### Project Structure
```bash
nova-ai-assistant/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── models/
│   └── llm.py             # LLM initialization and tools
├── prompts/
│   └── templates.py       # System prompt templates
├── utils/
│   ├── document_utils.py  # Document processing
│   ├── search_utils.py    # Web search functions
│   └── ui_utils.py        # Streamlit UI components
└── requirements.txt       # Dependencies
```

### Key Components
* Document Processing
    - Automatic file type detection
    - Chunk-based text splitting
    - In-memory vector store with Cohere/HuggingFace embeddings

* AI Pipeline
    - Dynamic prompt engineering
    - Automatic tool selection (web/doc search)
    - Thought process visualization

* UI Features
    - Dark/light mode support
    - Document preview pane
    - Interactive chat history
    - File-type specific visual indicators


### 🤝 Contributing
We welcome contributions! Please follow these steps:
    1. Fork the repository
    2. Create your feature branch (git checkout -b feature/AmazingFeature)
    3. Commit your changes (git commit -m 'Add some AmazingFeature')
    4. Push to the branch (git push origin feature/AmazingFeature)
    5. Open a Pull Request


### 📜 License
Distributed under the MIT License. See LICENSE for more information.

### 🌟 Acknowledgments
- Groq for ultra-fast LLM inference
- Tavily for real-time web search API
- LangChain team for document processing tools
- Streamlit for interactive UI framework
