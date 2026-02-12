from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="CS 5542 Demo API")


class EchoRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"status": "ok", "message": "API working"}


@app.post("/echo")
def echo(req: EchoRequest):
    return {"echo": req.text}
