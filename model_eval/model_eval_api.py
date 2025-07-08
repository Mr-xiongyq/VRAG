from fastapi import FastAPI, Request
import uvicorn
from typing import List, Dict, Any  
from model_eval import LLMGenerator
from tqdm import tqdm
import os
import json

app = FastAPI()
model_eval = None

@app.on_event("startup")
async def startup_event():
    global model_eval
    # model_name = "Qwen/Qwen2.5-72B-Instruct"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_eval = LLMGenerator(model_name)

@app.post("/eval")
async def eval(request: Request):
    prompts = await request.json()
    prompts = json.loads(prompts)
    bs = int(prompts['bs'])
    data_eval = prompts['prompts']
    eval_results = []
    for i in tqdm(range(0, len(data_eval), bs)):
        eval_results.extend(model_eval.eval_func(data_eval[i:i+bs]))
    return eval_results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)