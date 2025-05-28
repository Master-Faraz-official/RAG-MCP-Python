import os
from dotenv import load_dotenv
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.embeddings import SentenceTransformerEmbeddings


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "mcp_docs"

# Initialize embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Qdrant client and collection
client = QdrantClient(url=QDRANT_URL)
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# Create VectorStore interface
vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embedding_model)

def load_and_store_documents(file_path: str):
    loader = TextLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    vectorstore.add_documents(chunks)
    return len(chunks)

def retrieve_documents(query: str, k: int = 5):
    return vectorstore.similarity_search(query, k=k)
