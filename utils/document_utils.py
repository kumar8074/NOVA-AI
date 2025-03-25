import os
import streamlit as st
from io import BytesIO
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from config import CHUNK_SIZE, CHUNK_OVERLAP

def process_pdf_file(uploaded_file, vector_store):
    """Process a PDF file and add it to the vector store"""
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
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True
        )
        document_chunks = text_processor.split_documents(raw_docs)
        
        # Add metadata to track source document
        for chunk in document_chunks:
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = uploaded_file.name
        
        # Add to vector store
        vector_store.add_documents(document_chunks)
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        return len(document_chunks)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.error(f"Error processing PDF: {str(e)}")
        return 0

def query_documents(query: str) -> str:
    """Query the vector store for relevant document chunks"""
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
        
        return context_text
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

def needs_document_search(query):
    """Determine if a document search is needed based on the query"""
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