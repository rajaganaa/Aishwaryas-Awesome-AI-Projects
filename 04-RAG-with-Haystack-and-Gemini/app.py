import os
from dotenv import load_dotenv

# LangChain components for the Generator part
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- HAYSTACK IMPORTS (Final Corrected Version) ---
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# --- HAYSTACK SETUP FOR RETRIEVAL ---
print("Initializing In-Memory Document Store...")
document_store = InMemoryDocumentStore()
print("Document store initialized.")

print("Defining indexing pipeline...")
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", PyPDFToDocument())
indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=500, split_overlap=100))
indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

indexing_pipeline.connect("converter", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")
print("Indexing pipeline defined.")

pdf_folder = "./data"
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

if not pdf_files:
    print("No PDF files found in the 'data' folder. Please add a PDF to continue.")
else:
    print(f"Indexing {len(pdf_files)} PDF file(s)...")
    indexing_pipeline.run({"converter": {"sources": pdf_files}})
    print("Indexing complete. Document store is ready.")

    # --- SETUP FOR GENERATION (using LangChain for simplicity) ---
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
    template = """
    Answer the question based only on the following context:
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    
    # --- INTERACTIVE Q&A LOOP ---
    print("\nVector store is ready. Enter your questions. Type 'exit' to quit.")
    
    # Initialize the Text Embedder outside the loop for efficiency
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    text_embedder.warm_up()  # Warm up the model once
    
    while True:
        user_question = input("Question: ")
        if user_question.lower() == 'exit':
            break

        retrieved_docs_payload = retriever.run(
            query_embedding=text_embedder.run(text=user_question)["embedding"], top_k=3
        )
        retrieved_docs = retrieved_docs_payload.get("documents", [])

        if not retrieved_docs:
            print("\nAnswer: I could not find any relevant information in the documents to answer your question.")
            continue

        context_string = "\n\n".join([doc.content for doc in retrieved_docs])
        
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        response = llm_chain.run(question=user_question, context=context_string)

        print("\nAnswer:")
        print(response)
        print("\n" + "="*50 + "\n")