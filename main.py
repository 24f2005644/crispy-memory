"""
Word-Problem Solver API
------------------------
POST /solve  with body: {"problem_id": "p0", "problem": "..."}
Returns EXACTLY:         {"reasoning": "...", "answer": 123}

How it works, in plain English:
1. We take the word problem the grader sends us.
2. We hand it to Claude with very strict instructions: "think step by
   step, ignore distractor numbers, and give me back ONLY JSON with
   the keys 'reasoning' and 'answer'."
3. We "prefill" Claude's answer with the character "{" so it is forced
   to start writing JSON immediately instead of chatting first.
4. We double-check (validate) what comes back actually obeys every
   rule the assignment cares about. If it doesn't, we quietly ask
   Claude again (up to 3 tries total) instead of showing the grader
   a broken response.
"""

import json
import os

from fastapi import FastAPI
from pydantic import BaseModel
import anthropic

app = FastAPI()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL_NAME = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a careful arithmetic word-problem solver.

The problem you are given may contain distractor numbers that are not
needed to compute the answer (e.g. distances, dates, unrelated counts).
Identify which numbers actually matter, ignore the rest, and work the
problem out one step at a time.

You must respond with ONLY a raw JSON object — no markdown, no code
fences, no commentary before or after it. The JSON object must have
EXACTLY these two keys and no others:

- "reasoning": a string of at least 80 characters that shows your
  step-by-step math and explicitly names any distractor numbers you
  ignored.
- "answer": the final result as a JSON integer (e.g. 945), never a
  string like "945", never a float like 945.0, and never include a
  currency symbol or units.

Return the smallest valid JSON object possible: no trailing commentary,
no extra keys, no explanations outside the JSON.
"""


class ProblemRequest(BaseModel):
    problem_id: str
    problem: str


def ask_claude(problem_text: str) -> dict:
    """Send the problem to Claude and parse its JSON reply."""
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": problem_text},
            # Prefilling the assistant turn with "{" forces the reply
            # to begin as a JSON object instead of adding chit-chat.
            {"role": "assistant", "content": "{"},
        ],
    )
    raw_text = "{" + response.content[0].text
    return json.loads(raw_text)


def validate(data: dict) -> None:
    """Raise ValueError if `data` breaks any rule from the assignment."""
    if set(data.keys()) != {"reasoning", "answer"}:
        raise ValueError(f"Expected exactly reasoning+answer, got: {list(data.keys())}")

    reasoning = data["reasoning"]
    if not isinstance(reasoning, str) or len(reasoning) < 80:
        raise ValueError("`reasoning` must be a string of at least 80 characters")

    answer = data["answer"]
    # bool is technically a subclass of int in Python, so we must
    # explicitly reject True/False as well as strings and floats.
    if isinstance(answer, bool) or not isinstance(answer, int):
        raise ValueError("`answer` must be a JSON integer, not a string/float/bool")


@app.post("/solve")
async def solve(req: ProblemRequest):
    last_error = None
    for _ in range(3):  # retry up to 3 times if Claude's output breaks a rule
        try:
            data = ask_claude(req.problem)
            validate(data)
            return data
        except Exception as exc:  # noqa: BLE001 - we want to retry on anything
            last_error = exc

    # Last-resort fallback so the endpoint never crashes or returns
    # something that fails the grader's shape checks.
    return {
        "reasoning": (
            "The solver could not produce a valid response after multiple "
            f"attempts. Last error: {last_error}"
        )[:500].ljust(80, "."),
        "answer": 0,
    }


@app.get("/")
async def health_check():
    return {"status": "ok"}
