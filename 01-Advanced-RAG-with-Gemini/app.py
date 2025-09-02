import streamlit as st
import os
import time
import nest_asyncio  # <-- ADD THIS LINE
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

nest_asyncio.apply() # <-- AND ADD THIS LINE

# Load the environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# --- CORE LANGCHAIN LOGIC ---

# 1. Define the LLM (Chat Model) - We use Gemini Pro
# llm = ChatGoogleGenerativeAI(model="gemini-pro")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# 2. Define the Prompt Template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions:{input}
"""
)

# def create_vector_store():
#     """
#     This function creates the vector store from PDF documents.
#     It performs: Data Ingestion -> Chunk Creation -> Vector Embeddings -> Storage
#     """
#     # Initialize session state variables if they don't exist
#     if "vectors" not in st.session_state:
#         st.write("Creating vector store DB...")
        
#         # Use Google's embedding model
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
#         # Load PDF documents from the specified directory
#         st.session_state.loader = PyPDFDirectoryLoader("./us_census")
#         st.session_state.docs = st.session_state.loader.load()
        
#         # Split documents into smaller chunks
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
#         # Create the FAISS vector store from the documents and embeddings
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
#         st.write("Vector Store DB is ready.")

def create_vector_store():
    """
    This function creates the vector store from PDF documents.
    It performs: Data Ingestion -> Chunk Creation -> Vector Embeddings -> Storage
    """
    if "vectors" not in st.session_state:
        st.write("Creating vector store DB...")
        
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load PDF documents from the specified directory
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        
        if not st.session_state.docs:
            st.error("No PDF documents found in the 'us_census' folder. Please add a PDF and try again.")
            return 

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # --- THIS IS THE CORRECTED LINE ---
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.write("Vector Store DB is ready.")

# --- STREAMLIT UI ---

st.title("Advanced RAG with Gemini")

# Button to trigger the embedding process
if st.button("Create Vector Store"):
    create_vector_store()

# Text input for the user's question
prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1:
    if "vectors" in st.session_state:
        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Get the retriever from the vector store
        retriever = st.session_state.vectors.as_retriever()
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        
        # Invoke the chain with the user's input
        response = retrieval_chain.invoke({'input': prompt1})
        
        print("Response time:", time.process_time() - start)
        
        # Display the answer
        st.write(response['answer'])

        # With a streamlit expander to show the context documents
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.write("Please create the vector store first by clicking the button.")