from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse


VLLM_URL = "http://vllm:8000/v1/chat/completions"
MODEL_NAME = "scam-8b-sft"

SYSTEM_PROMPT = """You are a strict binary classification system specialized in fraud detection. Your task is to analyze a conversation log between two parties and determine if it exhibits characteristics of a scam or fraudulent intent.

**Input Format:**
The user will provide a conversation text enclosed within <conversation> tags.

**Classification Criteria:**
- Output 'True': If the conversation contains evidence of scamming, phishing, social engineering, financial fraud, or malicious intent by either party.
- Output 'False': If the conversation appears to be a normal, benign interaction without fraudulent intent.

**Output Constraints (CRITICAL):**
1. You must output EXACTLY one word: "True" or "False".
2. Do NOT output any explanation, reasoning, preamble, or punctuation.
3. Do NOT output markdown formatting (e.g., no bold, no code blocks).
4. Do NOT apologize or converse.
5. If the input is empty or unintelligible, output "False" (as the safe default) or handle strictly as per specific edge-case logic.
"""

USER_TEMPLATE = (
    'Here is the conversation log to classify: <conversation>\n'
    '{conversation}\n'
    '</conversation>\n'
    'Reminder: Based on the system instructions, output ONLY "True" or "False".'
)


class PredictIn(BaseModel):
    text: str  # 前端丟來的 conversation 文字


class PredictOut(BaseModel):
    output: str  # "True" or "False"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=30.0)
    yield
    await app.state.http.aclose()


app = FastAPI(lifespan=lifespan)



def _coerce_boolean_word(raw: str) -> str:
    # 嚴格只接受 True/False；不符合就回 False
    if raw is None:
        return "False"
    s = raw.strip()
    if s == "True":
        return "True"
    if s == "False":
        return "False"
    return "False"



@app.post("/predict", response_class=PlainTextResponse)
async def predict(inp: PredictIn):
    conversation = inp.text

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(conversation=conversation)},
        ],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }

    r = await app.state.http.post(VLLM_URL, json=payload)
    r.raise_for_status()
    data = r.json()

    raw_out = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    out = _coerce_boolean_word(raw_out)
    print(out)  # 只會印在後端 console
    return out  # 前端只會拿到 True 或 False（純文字）

