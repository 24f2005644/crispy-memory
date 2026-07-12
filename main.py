"""
Word-Problem Solver API
------------------------
POST /solve  with body: {"problem_id": "p0", "problem": "..."}
Returns EXACTLY:         {"reasoning": "...", "answer": 123}

This version talks to an LLM through AI Pipe (https://aipipe.org),
the OpenAI-compatible proxy IITM course infrastructure issues student
tokens for. AI Pipe forwards requests to OpenAI, so we use the
`openai` Python SDK pointed at AI Pipe's base URL instead of OpenAI's.

How it works, in plain English:
1. We take the word problem the grader sends us.
2. We hand it to the model with strict instructions: think step by
   step, ignore distractor numbers, and give back ONLY the keys
   'reasoning' and 'answer'.
3. We use OpenAI's "structured outputs" feature (response_format with
   a JSON schema) so the API itself guarantees the reply matches our
   schema shape and types, instead of hoping the model behaves.
4. We still double-check (validate) the result against every rule the
   assignment cares about (like the >=80 char reasoning length, which
   JSON-schema "strict" mode can't enforce on its own) and retry up to
   3 times if something is off, instead of ever crashing.
"""

import json
import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("solver")

app = FastAPI()

# AI Pipe token: get yours at https://aipipe.org/login
# Set this as the AIPIPE_TOKEN environment variable when you deploy.
client = OpenAI(
    api_key=os.environ["AIPIPE_TOKEN"],
    base_url="https://aipipe.org/openai/v1",
)

MODEL_NAME = "gpt-5-nano"  # any OpenAI model AI Pipe supports; swap for a stronger one if you have budget

SYSTEM_PROMPT = """You are a careful arithmetic word-problem solver.

The problem you are given may contain distractor numbers that are not
needed to compute the answer (e.g. distances, dates, unrelated counts).
Identify which numbers actually matter, ignore the rest, and work the
problem out one step at a time.

Respond with:
- "reasoning": a string of at least 80 characters that shows your
  step-by-step math and explicitly names any distractor numbers you
  ignored.
- "answer": the final result as an integer (e.g. 945), never a string
  like "945", never a float like 945.0, and never include a currency
  symbol or units.
"""

# JSON Schema used with OpenAI's structured-outputs feature. This makes
# the API itself enforce the two keys and their types, rather than
# just hoping the model follows plain-English instructions.
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "word_problem_solution",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "answer": {"type": "integer"},
            },
            "required": ["reasoning", "answer"],
            "additionalProperties": False,
        },
    },
}


class ProblemRequest(BaseModel):
    problem_id: str
    problem: str


def extract_first_json_object(text: str) -> str:
    """Return just the first balanced {...} block in `text`.

    Structured outputs should already return clean JSON, but this is
    kept as a defensive fallback in case a particular model/proxy path
    ever appends stray text after the object.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"No complete JSON object found in model output: {text!r}")


def ask_llm(problem_text: str) -> dict:
    """Send the problem to the model (via AI Pipe) and parse its JSON reply."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ],
        response_format=RESPONSE_SCHEMA,
    )
    raw_text = response.choices[0].message.content
    json_str = extract_first_json_object(raw_text)
    return json.loads(json_str)


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
    for attempt in range(1, 4):  # retry up to 3 times if the model's output breaks a rule
        try:
            data = ask_llm(req.problem)
            validate(data)
            return data
        except Exception as exc:  # noqa: BLE001 - we want to retry on anything
            last_error = exc
            logger.warning(
                "problem_id=%s attempt=%d failed: %s", req.problem_id, attempt, exc
            )

    # Last-resort fallback so the endpoint never crashes or returns
    # something that fails the grader's shape checks. Logged with the
    # problem_id so you can grep your server logs (e.g. Render's "Logs"
    # tab) to see exactly what went wrong.
    logger.error("problem_id=%s exhausted retries: %s", req.problem_id, last_error)
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
