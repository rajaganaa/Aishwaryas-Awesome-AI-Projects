import os
# import weaviate <- REMOVED
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
# from langchain_weaviate.vectorstores import Weaviate <- REMOVED
from langchain_community.vectorstores import Chroma # <-- NEW IMPORT
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# --- CORE LANGCHAIN LOGIC ---

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# --- SETUP FOR CHROMA + HUGGINGFACE ---

print("Initializing HuggingFace Embeddings model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Model initialized.")

print("Loading documents...")
loader = PyPDFDirectoryLoader("./data")
docs = loader.load()

if not docs:
    print("No PDF documents found in the 'data' folder. Please add a PDF and try again.")
else:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    print(f"Split document into {len(documents)} chunks.")

    # 6. Setup ChromaDB vector store (REPLACED WEAVIATE)
    print("Setting up ChromaDB Vector Store...")
    vector_store = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory="./chroma_db" # This will save the DB to a folder
    )
    print("Vector store is ready.")

    # 8. Define the RAG Chain
    retriever = vector_store.as_retriever()
    output_parser = StrOutputParser()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

    # --- INTERACTIVE Q&A LOOP ---
    print("\nEnter your questions. Type 'exit' to quit.")
    while True:
        user_question = input("Question: ")
        if user_question.lower() == 'exit':
            break
        
        response = rag_chain.invoke(user_question)
        
        print("\nAnswer:")
        print(response)
        print("\n" + "="*50 + "\n")