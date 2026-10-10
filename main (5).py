import io
import os
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()

# CORS is required for the grader
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# ---------- Part 1: tool function ----------
def execute_python_code(code: str) -> dict:
    """Run code, return {"success": bool, "output": exact stdout or traceback}."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            exec(code, {"__name__": "__main__"})
        return {"success": True, "output": buf.getvalue()}
    except BaseException:
        return {"success": False, "output": traceback.format_exc()}


# ---------- Part 2: AI error analysis (only on failure) ----------
def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    client = OpenAI(
        api_key=os.environ["AIPIPE_TOKEN"],
        base_url="https://aipipe.org/openrouter/v1",
    )
    numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(code.split("\n"), 1))
    prompt = f"""Analyze this Python code and its error traceback.
Identify the line number(s) of the code where the error occurred.
Line numbers are 1-based and match the numbering shown below.
Only use line numbers that appear in the traceback for File "<string>".

CODE (with line numbers):
{numbered}

TRACEBACK:
{tb}

Return the line number(s) where the error is located."""

    resp = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "error_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "error_lines": {"type": "array", "items": {"type": "integer"}}
                    },
                    "required": ["error_lines"],
                    "additionalProperties": False,
                },
            },
        },
        temperature=0,
    )
    return ErrorAnalysis.model_validate_json(resp.choices[0].message.content).error_lines


def lines_from_traceback(tb: str) -> List[int]:
    """Deterministic fallback if the AI call fails."""
    m = re.findall(r'File "<string>", line (\d+)', tb)
    return [int(m[-1])] if m else []


@app.post("/code-interpreter")
def code_interpreter(req: CodeRequest):
    result = execute_python_code(req.code)
    if result["success"]:
        return {"error": [], "result": result["output"]}
    try:
        lines = analyze_error_with_ai(req.code, result["output"])
    except Exception:
        lines = lines_from_traceback(result["output"])
    return {"error": lines, "result": result["output"]}
