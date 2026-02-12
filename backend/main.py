from fastapi import FastAPI
from typing import List, Dict, Any
from pydantic import BaseModel
from backend.rag_pipeline import run_hybrid_query

app = FastAPI(title="CS 5542 Demo API")


class EchoRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"status": "ok", "message": "API working"}


@app.post("/echo")
def echo(req: EchoRequest):
    return {"echo": req.text}




history = {}

def log_results(userID, results):
    if userID not in history:
        history[userID] = []
    history[userID] += results
    print(f"Total history for {userID}: {len(history[userID])} items") 


def run_query(userID, query, top_k, alpha):
    results = run_hybrid_query(question=query, top_k=top_k, alpha=alpha)
    log_results(userID=userID, results=results)
    return results
    


def generate_answer(query, evidence):
    # call llm
    print()

@app.get("/history")
def get_history(userID:str):
    return history[userID]

'''
# Explain BM25 length normalization. 

Input:
• question
• top_k
• retrieval_mode
• use_multimodal
Output:
• generated answer
• retrieved evidence list
• retrieval scores
• runtime metrics
• failure_flag

Grounding rule
If evidence is missing:
Not enough evidence in the retrieved context.
'''

@app.post("/query")
def query(userID:str, query:str, top_k:int, alpha:float) :
    print("]]]]]]]]]]]]]]]]]]]]]]]]]]]]")
    x = run_query(userID=userID, query=query, top_k=top_k, alpha=alpha)
    return history[userID]


