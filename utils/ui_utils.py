import streamlit as st

def apply_custom_styling():
    """Apply custom styling to the Streamlit app"""
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
    
    /* Document list styling */
    .document-list {
        margin-top: 10px;
    }
    
    .document-item {
        padding: 8px;
        margin: 4px 0;
        background-color: #1E1E1E;
        border-radius: 4px;
        border-left: 3px solid #00FFAA;
    }
    
    /* File type badges */
    .file-badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 6px;
    }
    
    .file-badge-pdf {
        background-color: #FF5252;
        color: white;
    }
    
    .file-badge-docx {
        background-color: #2196F3;
        color: white;
    }
    
    .file-badge-xlsx {
        background-color: #4CAF50;
        color: white;
    }
    
    .file-badge-txt {
        background-color: #9E9E9E;
        color: white;
    }
    
    /* Custom document expander */
    .document-expander {
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize the session state variables"""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "document_contents" not in st.session_state:
        st.session_state.document_contents = {}
    
    if "message_log" not in st.session_state:
        st.session_state.message_log = [
            {"role": "ai", "content": "Hi! I'm NOVA, your AI assistant with multi-document intelligence. What can I do for you today? You can chat with me or upload documents (PDF, DOCX, CSV, Excel, PPT, TXT, and more) for analysis."}
        ]
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "has_documents" not in st.session_state:
        st.session_state.has_documents = False
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
        
    if "uploaded_file_info" not in st.session_state:
        st.session_state.uploaded_file_info = {}
    
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False

def display_file_badge(file_type):
    """Generate HTML for a file type badge"""
    # Determine badge class based on file type
    badge_class = "file-badge"
    if file_type.lower() in ["pdf"]:
        badge_class += " file-badge-pdf"
    elif file_type.lower() in ["docx", "doc"]:
        badge_class += " file-badge-docx"
    elif file_type.lower() in ["xlsx", "xls", "csv"]:
        badge_class += " file-badge-xlsx"
    else:
        badge_class += " file-badge-txt"
        
    return f'<span class="{badge_class}">{file_type}</span>'