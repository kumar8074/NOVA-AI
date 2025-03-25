import streamlit as st
import re
import os
import datetime
from io import BytesIO
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Import from local modules
from config import CURRENT_DATE
from utils.ui_utils import apply_custom_styling, initialize_session_state
from utils.document_utils import process_pdf_file, query_documents, needs_document_search
from utils.search_utils import perform_web_search, needs_web_search
from models.llm import initialize_llm, initialize_embeddings, initialize_vector_store, build_prompt_chain
from prompts.templates import SYSTEM_TEMPLATE, SEARCH_INSTRUCTION_TEMPLATE, DOCUMENT_INSTRUCTION_TEMPLATE

# Apply custom styling
apply_custom_styling()

# Initialize session state
initialize_session_state()

# Set up the page
st.title("NOVA")
st.caption("Your AI Assistant with Document Intelligence")

# Initialize LLM and embeddings
llm_engine = initialize_llm()
embedding_model = initialize_embeddings()

# Initialize vector store if not already done
if st.session_state.vector_store is None:
    st.session_state.vector_store = initialize_vector_store(embedding_model)

# Create main chat container
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
        # Simple upload button
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
            num_chunks = process_pdf_file(uploaded_pdf, st.session_state.vector_store)
            
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

# Handle user input
if user_query:
    # Add user message to chat history immediately
    st.session_state.message_log.append({"role": "user", "content": user_query})
    st.session_state.processing = True
    st.rerun()

# Continue processing if in processing state
if st.session_state.processing:
    with st.spinner(""):
        messages = build_prompt_chain(st.session_state.message_log, SYSTEM_TEMPLATE)
        
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
                doc_instruction = DOCUMENT_INSTRUCTION_TEMPLATE.format(document_results=document_results)
                
                # Add the document results as context
                doc_context_message = SystemMessage(content=doc_instruction)
                
                # Create new messages list with the document context
                doc_messages = [
                    SystemMessage(content=SYSTEM_TEMPLATE),
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
            search_instruction = SEARCH_INSTRUCTION_TEMPLATE.format(search_results=search_results)
            
            # Add the search results as context
            web_context_message = SystemMessage(content=search_instruction)
            
            # Create new messages list with the search context
            search_messages = [
                SystemMessage(content=SYSTEM_TEMPLATE),
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