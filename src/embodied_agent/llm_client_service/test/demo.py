"""
Reference demo for llm_client_service.

Covers four usage patterns:
  1. Plain text call
  2. Structured JSON output (force_json)
  3. Tool / skill selection (function calling)
  4. Multi-turn conversation with caller-managed history

Run:
    python test/demo.py

Change MODEL to any key defined in llm_models.yaml.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_client import call_llm

MODEL = "qwen3-vl-32b-thinking"


# ── 1. Plain text call ────────────────────────────────────────────────────────
# Simplest usage: a system prompt and a single user prompt.
# The response dict always has the same shape regardless of model or scenario.

resp = call_llm(
    model=MODEL,
    system="You are a helpful robot assistant. Be concise.",
    prompt="What is the capital of France? Answer in one sentence.",
)

# status_code 200 means success; anything else means an error (see README).
if resp["status_code"] != 200:
    raise RuntimeError(f"[1] {resp['status_code']} {resp['message']}")

print("── 1. Plain text ──")
print(f"content  : {resp['content']}")
print(f"usage    : {resp['usage']}")  # prompt / completion / total tokens
print(f"timing   : {resp['timing']}")  # request_ms, total_ms
# resp["reasoning"] holds the model's chain-of-thought when available (e.g. DeepSeek-R1).
# resp["raw"]       holds the full provider response for debugging.


# ── 2. Structured JSON output ─────────────────────────────────────────────────
# force_json=True instructs the model to return only valid JSON.
# Always pair it with a system prompt that reinforces the constraint.

resp2 = call_llm(
    model=MODEL,
    system="Only output valid JSON. No extra text or markdown.",
    prompt='Return a JSON object with fields "city" and "country" for the capital of Japan.',
    force_json=True,
)

if resp2["status_code"] != 200:
    raise RuntimeError(f"[2] {resp2['status_code']} {resp2['message']}")
data = json.loads(resp2["content"])  # safe to parse directly when force_json=True

print("\n── 2. force_json ──")
print(f"parsed   : {data}")
print(f"timing   : {resp2['timing']}")


# ── 3. Tool / skill selection ─────────────────────────────────────────────────
# Pass robot skills as OpenAI-format tool definitions.
# The interface forwards them to the provider unchanged and returns tool_calls.
# Executing the selected skill is the caller's responsibility.

SKILLS = [
    {
        "type": "function",
        "function": {
            "name": "move_to_named_pose",
            "description": "Move the robot arm to a predefined named pose.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pose_name": {
                        "type": "string",
                        "description": "Target pose name, e.g. home, zero, observe_table",
                    }
                },
                "required": ["pose_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_gripper",
            "description": "Open the robot gripper fully.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

resp3 = call_llm(
    model=MODEL,
    system="You are a robot planner. Use the provided tools to fulfill the user request.",
    prompt="Move the robot arm back to the home position.",
    tools=SKILLS,
)

if resp3["status_code"] != 200:
    raise RuntimeError(f"[3] {resp3['status_code']} {resp3['message']}")

print("\n── 3. Tool / skill selection ──")
# When the model chooses a tool, tool_calls is non-empty and content is usually empty.
# When the model replies in plain text instead, tool_calls is [] and content has the reply.
for call in resp3["tool_calls"]:
    fn = call["function"]
    args = json.loads(fn["arguments"])
    print(f"skill    : {fn['name']}({args})")
if not resp3["tool_calls"]:
    print(f"content  : {resp3['content']}")  # model answered in text instead
print(f"timing   : {resp3['timing']}")


# ── 4. Multi-turn conversation ────────────────────────────────────────────────
# The interface is stateless — it never stores history internally.
# The caller maintains the history list and passes it on every turn.
# Each completed turn appends one user message and one assistant message.

history: list[dict] = []


def chat(user_message: str) -> str:
    """Send one turn, update history in place, return the assistant reply."""
    resp = call_llm(
        model=MODEL,
        system="You are a helpful robot assistant. Be concise.",
        messages=history,  # full history so far
        prompt=user_message,  # new user turn, appended internally before sending
    )
    if resp["status_code"] != 200:
        raise RuntimeError(f"{resp['status_code']}: {resp['message']}")
    # Manually append both sides so the next call sees the complete context.
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": resp["content"]})
    return resp["content"]


print("\n── 4. Multi-turn conversation ──")
print("User : What objects are typically found on a robot workspace table?")
print(f"Bot  : {chat('What objects are typically found on a robot workspace table?')}")
print("User : Which would be hardest for a robot arm to pick up?")
print(f"Bot  : {chat('Which would be hardest for a robot arm to pick up?')}")
print("User : How should the robot approach it safely?")
print(f"Bot  : {chat('How should the robot approach it safely?')}")
# At this point history holds 6 messages (3 user + 3 assistant).
