######### GROUND TRUTH FULL CODE BASE FOR NOVA #########

import os
import streamlit as st
import re
import datetime
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain_core.runnables import RunnableBranch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tavily import TavilyClient
from io import BytesIO

# Document processing imports
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

# Load environment variables
load_dotenv()

# Get API keys
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialize Tavily client
tavily_client = TavilyClient(api_key=tavily_api_key)

# Initialize embedding model
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")

# Apply custom styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    
    /* Style for upload button */
    .stButton button {
        background-color: #1E1E1E !important;
        color: #00FFAA !important;
        border: 1px solid #3A3A3A !important;
        border-radius: 5px;
        padding: 5px 10px;
        font-weight: bold;
        font-size: 16px;
    }
    
    .stButton button:hover {
        background-color: #2A2A2A !important;
        border: 1px solid #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("NOVA")
st.caption("Your AI Assistant with Document Intelligence")

# Initialize AI model (DeepSeek on Groq)
llm_engine = ChatGroq(model="Deepseek-R1-Distill-Qwen-32b", groq_api_key=groq_api_key)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore(embedding=EMBEDDING_MODEL)

if "document_contents" not in st.session_state:
    st.session_state.document_contents = {}

# Document processing functions
def process_pdf_file(uploaded_file):
    # Create a temporary file-like object
    pdf_file = BytesIO(uploaded_file.getvalue())
    
    # Save to a temporary file that PDFPlumberLoader can use
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(pdf_file.getvalue())
        
    # Use the regular PDFPlumberLoader
    loader = PDFPlumberLoader(temp_path)
    try:
        raw_docs = loader.load()
        
        # Store the raw document content for direct access
        full_text = "\n\n".join([doc.page_content for doc in raw_docs])
        st.session_state.document_contents[uploaded_file.name] = full_text
        
        # Chunk documents
        text_processor = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        document_chunks = text_processor.split_documents(raw_docs)
        
        # Add metadata to track source document
        for chunk in document_chunks:
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = uploaded_file.name
        
        # Add to vector store
        st.session_state.vector_store.add_documents(document_chunks)
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        return len(document_chunks)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.error(f"Error processing PDF: {str(e)}")
        return 0

# Define document RAG function with improved error handling and debugging
def query_documents(query: str) -> str:
    try:
        # Debug information
        doc_count = 0
        try:
            # This is a safer way to check document count that won't crash if structure changes
            if hasattr(st.session_state.vector_store, "_collection"):
                doc_count = st.session_state.vector_store._collection.count()
            else:
                # Alternative method if _collection doesn't exist
                doc_count = len(st.session_state.document_contents)
        except:
            doc_count = len(st.session_state.document_contents)
        
        # Check if there are documents in the vector store
        if doc_count == 0:
            return "No documents have been uploaded yet. Please upload a document first to enable document queries."
        
        # Find related documents
        try:
            relevant_docs = st.session_state.vector_store.similarity_search(query, k=4)
        except Exception as e:
            # Fallback to direct document search if vector search fails
            if len(st.session_state.document_contents) > 0:
                fallback_results = []
                for doc_name, content in st.session_state.document_contents.items():
                    fallback_results.append(f"Document: {doc_name}\nContent: {content[:1000]}...")
                
                context_text = "\n\n".join(fallback_results)
                
                # Log the issue but continue with fallback
                print(f"Vector search failed, using fallback: {str(e)}")
                
                return f"Vector search failed, using direct document content.\n\n{context_text}"
            else:
                return f"Error searching documents: {str(e)}"
        
        if not relevant_docs:
            # Fallback to direct document content search
            if len(st.session_state.document_contents) > 0:
                matching_docs = []
                for doc_name, content in st.session_state.document_contents.items():
                    # Simple keyword matching fallback
                    query_keywords = query.lower().split()
                    if any(keyword in content.lower() for keyword in query_keywords):
                        matching_docs.append(f"Document: {doc_name}\nContent: {content[:1000]}...")
                
                if matching_docs:
                    context_text = "\n\n".join(matching_docs)
                    return context_text
                else:
                    return "No relevant information found in the uploaded documents based on direct search."
            else:
                return "No relevant information found in the uploaded documents."
        
        # Create context from documents with source tracking
        doc_contexts = []
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get("source", f"Document {i+1}")
            doc_contexts.append(f"Document: {source}\nContent: {doc.page_content}")
        
        context_text = "\n\n".join(doc_contexts)
        
        # Document QA prompt
        doc_qa_prompt = """
        You are an expert research assistant. Use ONLY the provided document context to answer the query.
        If the answer is not in the provided context, state that you don't have the information.
        Be clear, factual, and provide specific references to the document parts you're using.
        
        Query: {query}
        Document Context: {context}
        
        Answer:
        """
        
        prompt = ChatPromptTemplate.from_template(doc_qa_prompt)
        response_chain = prompt | llm_engine
        
        response = response_chain.invoke({"query": query, "context": context_text})
        
        # Add source information
        result = response.content
        if not result.strip().lower().startswith("i don't have"):
            result += f"\n\n<think>Documents searched: {len(relevant_docs)} chunks from {', '.join(set([doc.metadata.get('source', 'Unknown') for doc in relevant_docs]))}</think>"
        
        return result
    except Exception as e:
        # Be more specific about the error and include debugging information
        error_message = f"Error querying documents: {str(e)}"
        print(error_message)  # For server logs
        
        # Include information about the document store state
        doc_info = "No document information available"
        if hasattr(st.session_state, "document_contents"):
            doc_names = list(st.session_state.document_contents.keys())
            doc_info = f"Available documents: {', '.join(doc_names) if doc_names else 'None'}"
        
        return f"{error_message}\n{doc_info}"

# Function to perform internet search
def perform_web_search(query: str) -> str:
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

# Get current date
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Updated system message template with emotion-related content removed
system_template = f"""
You are NOVA, an expert AI assistant with document analysis and web search capabilities, created by KUMAR.

Today's date is {current_date}.

Core capabilities:
1. Concise and accurate responses tailored to user needs
2. Real-time information retrieval via web search when necessary
3. Document analysis and question answering for uploaded PDF documents

When responding:
- If the user asks about current events, real-time data, or information you're uncertain about, use the web search results
- If the user asks about uploaded documents, use the document query results
- ALWAYS use web search or document query results when provided to give accurate information
- Summarize information from search or document results in your own words
- Cite sources by referring to "Source 1", "Source 2", etc. or "Document 1", "Document 2", etc and provide their relevant links.
- NEVER claim you can't access links, real-time data, or the internet
- When asked about dates, times, or current events, ALWAYS reference today's date: {current_date}
- If search or document results are provided BUT they're not relevant to the query, rely on your training instead
- If no search or document results are available, respond based on your training

Remember: For most conversational queries, you don't need web search or document queries. Use them only when necessary.

IMPORTANT DOCUMENT HANDLING INSTRUCTIONS:
- When users upload documents, you CAN access their content through the Document Query tool
- NEVER say you cannot access document content - you have direct access to all uploaded documents
- When answering document questions, cite specific parts of the document
- If a document doesn't contain relevant information, clearly state that instead of saying you can't access it
- If the document doesn't contain relevant information, use web search to answer the query, AND clearly state that you have used the websearch
"""

# Enhanced search system message
search_instruction_template = """
IMPORTANT: You have access to recent web search results about the user's query.

These search results may contain current information that you can use to answer the user's question.

Please follow these guidelines:
1. Evaluate if the search results are RELEVANT to the user's query
2. If relevant, synthesize information from the search results for an accurate, up-to-date answer
3. If NOT relevant, DO NOT use them and instead rely on your training
4. When using search results, cite your sources by referring to "Source 1", "Source 2", etc and provide their relevant links.
5. If the search results contain conflicting information, acknowledge this and present multiple perspectives
6. Present information in a clear, concise manner
7. NEVER claim you can't access real-time data or the internet

Here are the search results:

{search_results}
"""

# Document query system message
document_instruction_template = """
IMPORTANT: You have access to information from documents that the user has uploaded. The following content has been extracted from these documents.

Please follow these guidelines:
1. You CAN and SHOULD use this document information to answer the user's query
2. NEVER say you cannot access document content - you have direct access shown below
3. If the information below doesn't answer the query, state "The uploaded documents don't contain information about [specific topic]" 
4. If the uploaded documents does not contain information about [specific topic], use websearch to answer the query and provide relevant souce links
5. When using document information, cite your sources by referring to specific document names
6. Be specific about what parts of the documents you're using
7. Present information in a clear, concise manner
8. Do not make up information that isn't in the documents

Here is the document information:

{document_results}
"""

# Initialize message log in session state
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "ai", "content": "Hi! I'm NOVA, your AI assistant with document intelligence. What can I do for you today? You can chat with me or upload PDF documents for analysis."}
    ]

# Flag for processing state
if "processing" not in st.session_state:
    st.session_state.processing = False

# Flag for uploaded documents
if "has_documents" not in st.session_state:
    st.session_state.has_documents = False

# Store uploaded file names
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Flag to avoid duplicate document upload messages
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# Flag to control file uploader visibility
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

chat_container = st.container()

# Display previous messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            content = message["content"]
            think_matches = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
            content_without_think = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

            st.markdown(content_without_think)

            for think_text in think_matches:
                with st.expander("💭 Thought Process"):
                    st.markdown(think_text)
    
    # Show processing indicator only when processing
    if st.session_state.processing:
        with st.chat_message("ai"):
            st.write("Processing...")

# Display uploaded documents if any
if st.session_state.uploaded_files:
    with st.expander("📚 Uploaded Documents"):
        for file_name in st.session_state.uploaded_files:
            st.write(f"- {file_name}")

# Create a container for the upload button and chat input
input_container = st.container()

with input_container:
    # Create two columns for the chat input and upload button
    col1, col2 = st.columns([10, 2])
    
    with col2:
        # Simple upload button instead of custom HTML/JS
        upload_button = st.button("📄 +", key="upload_button")
        if upload_button:
            st.session_state.show_uploader = not st.session_state.show_uploader
    
    with col1:
        # Chat input
        user_query = st.chat_input("Ask NOVA...")

# Show the file uploader when toggled
if st.session_state.show_uploader:
    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type="pdf",
        key="pdf_uploader"
    )
    
    # Process uploaded file
    if uploaded_pdf and (st.session_state.last_uploaded_file != uploaded_pdf.name):
        with st.spinner("Processing document..."):
            num_chunks = process_pdf_file(uploaded_pdf)
            
            if num_chunks > 0:
                # Update state
                st.session_state.has_documents = True
                if uploaded_pdf.name not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(uploaded_pdf.name)
                
                # Add system message about the upload
                upload_message = f"📄 Document '{uploaded_pdf.name}' successfully uploaded and processed ({num_chunks} chunks). You can now ask questions about this document."
                st.session_state.message_log.append({"role": "ai", "content": upload_message})
                
                # Update last uploaded file to prevent duplicate messages
                st.session_state.last_uploaded_file = uploaded_pdf.name
                
                # Hide the uploader after successful upload
                st.session_state.show_uploader = False
                
                # Rerun to update UI
                st.rerun()
            else:
                st.error(f"Failed to process document '{uploaded_pdf.name}'. Please try again or use a different document.")

# Function to build the prompt chain
def build_prompt_chain():
    # Start with just the system message
    messages = [SystemMessage(content=system_template)]
    
    # Add the conversation history
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
    
    return messages

# Function to determine if web search is needed (emotion-related patterns removed)
def needs_web_search(query):
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

# Function to determine if we should check documents (simplified)
def needs_document_search(query):
    # Check for explicit request for document search
    document_patterns = [
        "in the document", "from the pdf", "in the pdf", "document says",
        "check the document", "in the uploaded", "from the uploaded",
        "the document mentions", "in my document", "in my pdf",
        "what does the document say about", "find in document",
        "tell me about the document", "summarize the document",
        "what's in the pdf", "what is in the document",
        "analyze the pdf", "analyze the document"
    ]
    
    query_lower = query.lower()
    
    for pattern in document_patterns:
        if pattern in query_lower:
            return True
    
    # If there are documents and the query sounds like it needs information:
    if st.session_state.has_documents and len(st.session_state.uploaded_files) > 0:
        info_patterns = [
            "what is", "how does", "tell me about", "explain", "summarize",
            "what are", "where is", "who is", "when did", "why did",
            "what was", "how many", "how much"
        ]
        
        for pattern in info_patterns:
            if query_lower.startswith(pattern):
                return True
    
    # Always try document search if we have few documents
    if st.session_state.has_documents and len(st.session_state.uploaded_files) <= 3:
        return True
    
    # Default to not using document search unless explicitly requested
    return False

# Handle user input
if user_query:
    # Add user message to chat history immediately
    st.session_state.message_log.append({"role": "user", "content": user_query})
    st.session_state.processing = True
    st.rerun()

# Continue processing if in processing state
if st.session_state.processing:
    with st.spinner(""):
        messages = build_prompt_chain()
        
        # Get the last user query
        last_user_query = st.session_state.message_log[-1]["content"]
        
        # Determine if web search or document search is needed
        should_search_web = needs_web_search(last_user_query)
        should_search_docs = needs_document_search(last_user_query)
        
        # Flag to track if we've already handled the query
        query_handled = False
        
        # Always try document search first if we have documents
        if st.session_state.has_documents:
            # Query documents
            document_results = query_documents(last_user_query)
            
            if "No documents have been uploaded yet" not in document_results and "Error" not in document_results:
                # Format document instruction with results
                doc_instruction = document_instruction_template.format(document_results=document_results)
                
                # Add the document results as context
                doc_context_message = SystemMessage(content=doc_instruction)
                
                # Create new messages list with the document context
                doc_messages = [
                    SystemMessage(content=system_template),
                    doc_context_message,
                    HumanMessage(content=f"{last_user_query}")
                ]
                
                # Get response from LLM with document results
                ai_response = llm_engine.invoke(doc_messages).content
                query_handled = True
                
                # Add a thought process about document search
                if "<think>" not in ai_response:
                    ai_response += f"\n\n<think>Document search was performed and used to generate this response.</think>"
        
        # Handle web search if needed and not already handled
        if should_search_web and not query_handled:
            # Perform web search
            search_results = perform_web_search(last_user_query)
            
            # Format search instruction with results
            search_instruction = search_instruction_template.format(search_results=search_results)
            
            # Add the search results as context
            web_context_message = SystemMessage(content=search_instruction)
            
            # Create new messages list with the search context
            search_messages = [
                SystemMessage(content=system_template),
                web_context_message,
                HumanMessage(content=f"{last_user_query}")
            ]
            
            # Get response from LLM with search results
            ai_response = llm_engine.invoke(search_messages).content
            query_handled = True
            
            # Add a thought process about web search
            if "<think>" not in ai_response:
                ai_response += f"\n\n<think>Web search was performed and used to generate this response.</think>"
        
        # If neither search was used or they didn't provide useful results
        if not query_handled:
            # Add the current user query to the messages
            messages.append(HumanMessage(content=last_user_query))
            
            # Use LLM directly
            ai_response = llm_engine.invoke(messages).content
            
            # Add a thought process about using base knowledge
            if "<think>" not in ai_response:
                ai_response += f"\n\n<think>No external search was performed. Response generated from base knowledge.</think>"

    # Add AI response to chat history
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Turn off processing state
    st.session_state.processing = False
    
    # Rerun to update the UI
    st.rerun()