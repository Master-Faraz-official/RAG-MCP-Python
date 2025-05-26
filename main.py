from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from lm_client import query_lm

app = FastAPI()

class PromptRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "lmstudio"
    temperature: float = 0.7
    max_tokens: int = 300

@app.post("/query")
def handle_query(req: PromptRequest):
    response = query_lm(req.messages, req.model, req.max_tokens, req.temperature)
    return {"response": response}
