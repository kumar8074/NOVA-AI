# Updated document_utils.py
import os
import streamlit as st
from io import BytesIO
import tempfile
import pandas as pd
import json
import yaml
import mammoth
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    PyMuPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from config import CHUNK_SIZE, CHUNK_OVERLAP

def get_file_extension(filename):
    """Get the file extension from a filename."""
    return os.path.splitext(filename)[1].lower()

def process_document_file(uploaded_file, vector_store):
    """Process various document file types and add them to the vector store"""
    # Get file extension
    file_extension = get_file_extension(uploaded_file.name)
    
    # Create a temporary file-like object
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    try:
        # Write the file to disk temporarily
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        # Process based on file type
        raw_docs = []
        
        # PDF Files
        if file_extension in ['.pdf']:
            loader = PDFPlumberLoader(temp_path)
            raw_docs = loader.load()
            
        # Text Files
        elif file_extension in ['.txt', '.md', '.log']:
            loader = TextLoader(temp_path)
            raw_docs = loader.load()
            
        # Microsoft Word Documents
        elif file_extension in ['.docx']:
            loader = Docx2txtLoader(temp_path)
            raw_docs = loader.load()
            
        # Legacy Word Documents - requires extra handling
        elif file_extension in ['.doc']:
            # Convert DOC to DOCX-like format using mammoth
            with open(temp_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
                text = result.value
            
            # Create a document directly
            from langchain_core.documents import Document
            raw_docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]
            
        # CSV Files
        elif file_extension in ['.csv']:
            loader = CSVLoader(temp_path)
            raw_docs = loader.load()
            
        # Excel Files
        elif file_extension in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(temp_path)
            raw_docs = loader.load()
            
        # PowerPoint Files
        elif file_extension in ['.ppt', '.pptx']:
            loader = UnstructuredPowerPointLoader(temp_path)
            raw_docs = loader.load()
            
        # JSON Files
        elif file_extension in ['.json']:
            # For JSON files, use a simple approach that works with various structures
            with open(temp_path, 'r') as file:
                json_data = json.load(file)
            
            # Convert to string representation for simple processing
            text = json.dumps(json_data, indent=2)
            from langchain_core.documents import Document
            raw_docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]
            
        # YAML Files
        elif file_extension in ['.yaml', '.yml']:
            # Similar approach for YAML files
            with open(temp_path, 'r') as file:
                yaml_data = yaml.safe_load(file)
            
            # Convert to string representation
            text = yaml.dump(yaml_data, default_flow_style=False)
            from langchain_core.documents import Document
            raw_docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]
            
        # HTML Files
        elif file_extension in ['.html', '.htm']:
            loader = UnstructuredHTMLLoader(temp_path)
            raw_docs = loader.load()
            
        else:
            # For unsupported file types, try a generic text loader as fallback
            try:
                loader = TextLoader(temp_path)
                raw_docs = loader.load()
            except Exception as e:
                raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Store the raw document content for direct access
        full_text = "\n\n".join([doc.page_content for doc in raw_docs])
        st.session_state.document_contents[uploaded_file.name] = full_text
        
        # Get document metadata to display to user
        file_size = os.path.getsize(temp_path) / 1024  # size in KB
        num_pages = len(raw_docs)
        
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
            
            # Add file type for better context
            chunk.metadata["file_type"] = file_extension.replace('.', '')
        
        # Add to vector store
        vector_store.add_documents(document_chunks)
        
        # Return processing stats
        return {
            "chunks": len(document_chunks),
            "file_size_kb": round(file_size, 2),
            "pages_or_sections": num_pages,
            "file_type": file_extension.replace('.', '').upper()
        }
        
    except Exception as e:
        raise Exception(f"Error processing document: {str(e)}")
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        try:
            os.rmdir(temp_dir)
        except:
            pass  # Directory might not be empty if other temp files were created

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
        
        # Create context from documents with source tracking and file type
        doc_contexts = []
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get("source", f"Document {i+1}")
            file_type = doc.metadata.get("file_type", "unknown").upper()
            doc_contexts.append(f"Document: {source} (Type: {file_type})\nContent: {doc.page_content}")
        
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
        "in the document", "from the file", "in the file", "document says",
        "check the document", "in the uploaded", "from the uploaded",
        "the document mentions", "in my document", "in my file",
        "what does the document say about", "find in document",
        "tell me about the document", "summarize the document",
        "what's in the file", "what is in the document",
        "analyze the file", "analyze the document"
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

def get_supported_file_types():
    """Return a list of supported file extensions"""
    return [
        "pdf", "txt", "md", "docx", "doc", "csv", "xlsx", 
        "xls", "ppt", "pptx", "json", "yaml", "yml", "html", "htm", "log"
    ]