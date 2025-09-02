import streamlit as st
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nest_asyncio
from dotenv import load_dotenv

# Apply the asyncio patch for Streamlit compatibility
nest_asyncio.apply()

# Load the environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# --- CORE LANGCHAIN LOGIC (ADAPTED FOR GEMINI) ---

# Define the LLM (Chat Model) using Gemini Pro
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Define the Prompt Template
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

def create_vector_store_from_url(url):
    """
    This function creates a FAISS vector store from a website URL.
    It handles loading, splitting, embedding, and storing the data.
    """
    if "vectors" not in st.session_state or st.session_state.url != url:
        st.write("Creating new vector store from URL...")
        st.session_state.url = url  # Store the URL to avoid re-creating
        
        # 1. Load Documents from the web
        loader = WebBaseLoader(url)
        docs = loader.load()

        if not docs:
            st.error("Could not load any content from the URL. Please check the URL and try again.")
            return

        # 2. Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        # 3. Create embeddings using Google's model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # 4. Create the FAISS vector store and store it in the session
        st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)
        st.write("Vector Store DB is ready.")
    else:
        st.write("Using existing vector store.")

# --- STREAMLIT UI ---

st.title("RAG From Website with Gemini")

# Input for the website URL
url_input = st.text_input("Enter a Website URL to create a knowledge base")

# Button to trigger the embedding process
if st.button("Process URL"):
    if url_input:
        create_vector_store_from_url(url_input)
    else:
        st.warning("Please enter a URL.")

# Text input for the user's question
prompt1 = st.text_input("Ask a question about the website content:")

if prompt1:
    if "vectors" in st.session_state:
        # Create the full RAG chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time:", time.process_time() - start)
        
        st.write("### Answer")
        st.write(response['answer'])

        # Expander to show the source documents
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.warning("Please process a URL first.")