from fastapi import FastAPI,UploadFile, File
from pydantic import BaseModel
from typing import List, Dict
from lm_client import query_lm,query_lm_from_text
from rag_pipeline import load_and_store_documents, retrieve_documents

app = FastAPI()

# class PromptRequest(BaseModel):
#     messages: List[Dict[str, str]]
#     model: str = "lmstudio"
#     temperature: float = 0.7
#     max_tokens: int = 300

# @app.post("/query")
# def handle_query(req: PromptRequest):
#     response = query_lm(req.messages, req.model, req.max_tokens, req.temperature)
#     return {"response": response}


class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    with open(file.filename, "wb") as f:
        f.write(await file.read())
    chunks = load_and_store_documents(file.filename)
    return {"message": f"{chunks} chunks added to vector store."}

@app.post("/query")
def ask_question(req: QueryRequest):
    docs = retrieve_documents(req.query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {req.query}"
    response = query_lm_from_text(prompt)
    return {"response": response}

