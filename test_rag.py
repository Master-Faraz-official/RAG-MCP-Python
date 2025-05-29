# test_rag.py

from rag_pipeline import load_and_store_documents, retrieve_documents

# Step 1: Load and store the document
num_chunks = load_and_store_documents("test_file.txt")
print(f"Stored {num_chunks} chunks into the vector DB.")

# Step 2: Try a query
query = "What is LangChain used for?"
results = retrieve_documents(query)

print("\nTop retrieved documents:")
for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:\n{doc.page_content}")
