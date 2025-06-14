import streamlit as st
import warnings
from backend import process_file_and_get_query_engine
import hashlib

# Suppress PyTorch/Streamlit compatibility warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set page configuration
st.set_page_config(page_title="Dynamic Document Query App", page_icon="📄", layout="wide")

# Streamlit app title
st.title("Dynamic Document Query Application")
st.markdown("Upload a PDF file and enter a query to search the document using a vector database.")

# Initialize session state
if 'response' not in st.session_state:
    st.session_state.response = ""
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None

# Cache the query engine creation
@st.cache_resource
def get_cached_query_engine(file_content, file_name):
    """
    Cache the query engine for a given file.
    
    Args:
        file_content (bytes): Content of the uploaded file.
        file_name (str): Name of the uploaded file.
        
    Returns:
        QueryEngine: LlamaIndex query engine.
    """
    return process_file_and_get_query_engine(file_content, file_name)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Process uploaded file
if uploaded_file is not None:
    try:
        # Read file content and compute hash
        file_content = uploaded_file.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        uploaded_file.seek(0)  # Reset file pointer
        file_name = uploaded_file.name

        # Check if file has changed
        if st.session_state.file_hash != file_hash:
            st.session_state.file_hash = file_hash
            st.session_state.response = ""  # Clear previous response
            # Build new query engine
            st.session_state.query_engine = get_cached_query_engine(file_content, file_name)
        else:
            # Reuse cached query engine
            query_engine = st.session_state.query_engine

        # Input form for query
        with st.form(key="query_form"):
            query = st.text_input("Enter your query:", value=st.session_state.query)
            submit_button = st.form_submit_button("Submit Query")

            # Process query
            if submit_button and query and query_engine:
                try:
                    st.session_state.query = query
                    response = query_engine.query(query)
                    st.session_state.response = str(response)
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.session_state.response = ""

        # Display response
        if st.session_state.response:
            st.subheader("Response")
            st.write(st.session_state.response)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a PDF file to proceed.")
    st.session_state.file_hash = None
    st.session_state.response = ""
    st.session_state.query_engine = None

# Display sample queries
st.subheader("Sample Queries")
