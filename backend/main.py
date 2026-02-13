from fastapi import FastAPI
from typing import List, Dict, Any
from pydantic import BaseModel
from backend.rag_pipeline import run_hybrid_query
from backend.json_to_csv import process_json_to_csv
import json

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

def save_history(filename='./data/data/history.json'):
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)

def get_history(userID:str):
    try:
        with open('./data/data/history.json', 'r') as f:
            history = json.load(f)
        return history.get(userID, [])
    except FileNotFoundError:
        return []
    

def log_results(userID, results):
    if userID not in history:
        history[userID] = []    
    history[userID] += [results]
    save_history()


def run_query(userID, query, top_k, retrieval_mode, alpha):
    results = run_hybrid_query(question=query, top_k=top_k, retrieval_mode=retrieval_mode, alpha=alpha)
    log_results(userID=userID, results=results)
    process_json_to_csv(results)
    return results
    


@app.get("/history")
def history_hook(userID:str) -> list[Dict[str,Any]]:
    return get_history(userID=userID)

@app.post("/query")
def query(query:str, top_k:int, retrieval_mode:str, alpha:float, userID:str="null") :
    return run_query(userID=userID, query=query, top_k=top_k, retrieval_mode=retrieval_mode, alpha=alpha)