import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import QdrantVectorStore

from langchain_community.vectorstores import Qdrant


# Load environment variables from .env
load_dotenv()

def get_env_variable(key, default=None):
    value = os.getenv(key)
    if value is None and default is None:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value or default

# Configuration
QDRANT_URL = get_env_variable("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "mcp_docs"

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Qdrant client
client = QdrantClient(url=QDRANT_URL)

# Create collection if it doesn't exist
existing_collections = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in existing_collections:
    print(f"Creating Qdrant collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
else:
    print(f"Using existing Qdrant collection '{COLLECTION_NAME}'.")

# Helper to get vector store instance
# def get_vectorstore():
#     return Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embedding_model)

# vectorstore = get_vectorstore()
# # Initialize vector store

# vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embeddings=embedding_model)
# vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embedding_model)
vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embedding_model)
# ...existing code...

# Load, split, and store documents into Qdrant
def load_and_store_documents(file_path: str):
    print(f"Loading file: {file_path}")
    loader = TextLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    vectorstore.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks into Qdrant collection: '{COLLECTION_NAME}'.")

    return len(chunks)

# Retrieve top-K similar documents for a given query
def retrieve_documents(query: str, k: int = 5):
    print(f"Retrieving top {k} documents for query: '{query}'")
    return vectorstore.similarity_search(query, k=k)
