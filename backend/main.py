from fastapi import FastAPI
from typing import List, Dict, Any
from pydantic import BaseModel
from backend.rag_pipeline import run_hybrid_query
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
    # return history[userID] '''<= uncomment if removing history.json and/or we want to read 
    #                              only the local/current session. [feb 12 2026]''' 
    try:
        with open('./data/data/history.json', 'r') as f:
            history = json.load(f)
        return history.get(userID, [])
    except FileNotFoundError:
        return []
    

'''
**E. Automatic Logging (Required)**
Each query must append one row to:
logs/query_metrics.csv
If the file does not exist, the application must create it automatically.
Columns:

TODO: - timestamp
- query_id
TODO: - retrieval_mode
- top_k
- latency_ms
TODO: - Precision@
TODO: - Recall@
- evidence_ids_returned
- faithfulness_pass
- missing_evidence_behavior
'''
def write_query_metrics(results):
    print()


def log_results(userID, results):
    if userID not in history:
        history[userID] = []    
    history[userID] += [results]
    save_history()

def run_query(userID, query, top_k, alpha):
    results = run_hybrid_query(question=query, top_k=top_k, alpha=alpha)
    log_results(userID=userID, results=results)
    return results
    


def generate_answer(query, evidence):
    ''' TODO: call llm to format answer. '''
    print()



'''
Input:
• question
• top_k
• TODO: retrieval_mode [REQUIRED : >=2 retrieval modes]
• TODO: use_multimodal
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

@app.get("/history")
def history(userID:str) -> list[Dict[str,Any]]:
    return get_history(userID=userID)

@app.post("/query")
def query(query:str, top_k:int, alpha:float, userID:str="null") :
    return run_query(userID=userID, query=query, top_k=top_k, alpha=alpha)