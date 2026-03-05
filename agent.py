import requests
import json
import os
from datetime import datetime
import time
from rag_engine import load_documents, build_index, search, format_context
from weather_tool import get_weather
from observability import AgentTrace, record_metric, print_metrics_summary, setup_logger

logger = setup_logger("agent")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
MEMORY_FILE = "agent_memory.json"

# ----------------------
# RAG SETUP — runs once when agent starts
# ----------------------
DOCUMENTS = load_documents("knowledge_base.txt")
VECTORIZER, MATRIX = build_index(DOCUMENTS)
# We build the index once at startup, not on every search.
# Building is expensive. Searching is cheap.
# This is the same pattern used in production search engines.


# ----------------------
# MEMORY SYSTEM
# ----------------------

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"sessions": [], "facts": [], "goals_completed": []}


def save_memory(memory: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def log_session(memory: dict, goal: str, steps: list, outcome: str):
    session = {
        "timestamp": datetime.now().isoformat(),
        "goal": goal,
        "steps_taken": steps,
        "outcome": outcome
    }
    memory["sessions"].append(session)
    save_memory(memory)


def get_recent_context(memory: dict, n: int = 3) -> str:
    recent = memory["sessions"][-n:]
    if not recent:
        return "No previous sessions."
    lines = []
    for s in recent:
        lines.append(f"- [{s['timestamp']}] Goal: {s['goal']} → {s['outcome']}")
    return "\n".join(lines)


# ----------------------
# REAL LLM CALL
# ----------------------

def call_llm(prompt):
    start = time.time()
    # record when we started

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()["response"]

    duration_ms = round((time.time() - start) * 1000)
    # calculate how long the call took in milliseconds

    record_metric("llm_call", {
        "duration_ms": duration_ms,
        "prompt_length": len(prompt),
        "response_length": len(result)
    })

    return result


# ----------------------
# VALIDATOR
# ----------------------

def is_valid_decision(decision: dict) -> bool:
    required_keys = {"thought", "action", "input"}
    allowed_actions = {"weather_api", "calculator", "final_answer", "knowledge_base"}

    if not required_keys.issubset(decision.keys()):
        return False

    if decision["action"] not in allowed_actions:
        return False

    if decision["action"] == "final_answer" and not decision["input"].strip():
        return False

    return True


# ----------------------
# REPETITION CHECK
# ----------------------

def is_repeated(decision, history):
    for past_step in history:
        if (past_step["decision"]["action"] == decision["action"] and
                past_step["decision"]["input"] == decision["input"]):
            return True
    return False

# ----------------------
# the LLM Planner — separate from the main think function
# ----------------------

def llm_plan(goal: str, past_context: str) -> list:
    """
    Separate LLM call that only produces a plan.
    Returns a list of steps to execute.
    We separate planning from execution because mixing them
    makes the LLM try to do both at once, which it does poorly.
    """
    prompt = f"""
You are a planning agent. Your ONLY job is to break a goal into steps.

PAST SESSIONS (learn from these, do not repeat mistakes):
{past_context}

GOAL:
{goal}

Analyze the goal and create an ordered list of steps to complete it.
Each step must use exactly one tool.

Available tools:
- weather_api     → ONLY for weather/temperature/climate questions. input = city name only.
- calculator      → ONLY for math calculations. input = pure math expression like "15000 + 96000"
- knowledge_base  → ONLY for policy, business, visa, banking, holiday questions. input = search query

Respond ONLY with a JSON array, no text before or after:
[
  {{"step": 1, "tool": "tool_name", "input": "input for that tool", "reason": "why this step"}},
  {{"step": 2, "tool": "tool_name", "input": "input for that tool", "reason": "why this step"}}
]

RULES:
1. Only include steps that are necessary for the goal.
2. calculator input must be a pure math expression only. Extract the numbers from the goal.
3. Maximum 5 steps.
4. Each step must be independent — do not reference results of previous steps in the input.
5. weather_api is the ONLY tool for weather questions. NEVER use knowledge_base for weather.
6. knowledge_base is ONLY for policy, regulations, business, visa questions — not weather.
"""

    reply = call_llm(prompt)

    try:
        # LLMs sometimes wrap JSON in markdown code blocks like ```json ... ```
        # We strip those out before parsing
        clean = reply.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except:
        print(f"  ❌ Plan parsing failed. Raw reply: {reply[:200]}")
        return []

# ----------------------
# Plan Validator — checks that the plan structure is correct before trusting it
# ----------------------

def is_valid_plan(plan: list) -> bool:
    """
    Checks that the plan is a non-empty list of valid steps.
    Same principle as is_valid_decision — validate before trusting.
    """
    if not isinstance(plan, list) or len(plan) == 0:
        return False

    allowed_tools = {"weather_api", "calculator", "knowledge_base"}

    for step in plan:
        if not all(k in step for k in ["step", "tool", "input"]):
            return False
        if step["tool"] not in allowed_tools:
            return False

    return True

# ----------------------
# Research Thinker — separate from the main think function, with its own prompt and logic
# ----------------------

def research_think(goal: str, findings: list, past_context: str) -> dict:
    """
    Unlike llm_think which decides one action, research_think
    evaluates ALL findings so far and decides:
    - what to search next, OR
    - that it has enough to answer
    
    findings is a list of {query, result} dicts — everything found so far.
    """

    findings_text = ""
    if findings:
        for i, f in enumerate(findings):
            findings_text += f"\nSearch {i+1}: '{f['query']}'\nResult: {f['result']}\n"
    else:
        findings_text = "No searches done yet."

    prompt = f"""
You are a research agent. Your job is to investigate a goal thoroughly.

PAST SESSIONS:
{past_context}

RESEARCH GOAL:
{goal}

FINDINGS SO FAR:
{findings_text}

Decide what to do next.

If you need more information → search the knowledge base with a NEW query not used before.
If you have enough information to answer confidently → write the final answer.

Respond ONLY in valid JSON:
{{"decision": "search" or "answer", "query": "search query if decision is search, else empty", "confidence": 0.0-1.0, "reasoning": "why you made this decision", "answer": "full answer if decision is answer, else empty"}}

RULES:
1. confidence is a number from 0.0 (no idea) to 1.0 (completely certain).
2. If confidence >= 0.7 and you have at least 1 finding, choose "answer".
3. If you have done 3 or more searches, you MUST choose "answer" regardless of confidence.
4. Each search query must be different from all previous queries.
5. answer must be a detailed paragraph, not a single sentence.
"""

    reply = call_llm(prompt)

    try:
        clean = reply.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        result = json.loads(clean.strip())
        return result
    except:
        return {
            "decision": "answer",
            "query": "",
            "confidence": 0.5,
            "reasoning": "parse error",
            "answer": reply
        }

# ----------------------
# The Research Agent Loop — uses research_think instead of llm_think, and has its own logic for when to stop searching and answer
# ----------------------

def run_research_agent(goal: str):
    """
    A separate agent runner specifically for open-ended research goals.
    Uses iterative search with confidence gating instead of fixed plans.
    """

    memory = load_memory()
    past_context = get_recent_context(memory)
    print(f"\n📂 Memory loaded. Past sessions found: {len(memory['sessions'])}")
    print(f"\n🔬 Starting research: {goal}\n")

    findings = []
    # findings grows as the agent discovers information
    # Each item: {"query": "...", "result": "..."}

    steps_log = []
    MAX_SEARCHES = 4
    # Hard cap — even if the agent never reaches confidence 0.7,
    # we stop at 4 searches to prevent runaway behavior

    for iteration in range(MAX_SEARCHES + 1):
        # +1 because the last iteration might be the answer step

        print(f"\n--- Research Iteration {iteration + 1} ---")

        decision = research_think(goal, findings, past_context)

        print(f"   Decision: {decision.get('decision')} | Confidence: {decision.get('confidence')}")
        print(f"   Reasoning: {decision.get('reasoning')}")

        if decision.get("decision") == "answer" or iteration == MAX_SEARCHES:
            # Either the agent is confident enough, or we hit the hard limit
            final_answer = decision.get("answer", "")

            if not final_answer and findings:
                # Safety net: if answer is empty but we have findings,
                # ask the LLM to synthesize what we found
                print("   ⚠️ Empty answer — running synthesis fallback")
                synthesis_prompt = f"""
Based on these research findings, write a detailed answer to: {goal}

FINDINGS:
{chr(10).join([f"- {f['query']}: {f['result']}" for f in findings])}

Write a clear, complete recommendation in plain English.
"""
                final_answer = call_llm(synthesis_prompt)

            print(f"\n✅ RESEARCH COMPLETE\n")
            print(f"Final Answer:\n{final_answer}")

            log_session(
                memory=memory,
                goal=goal,
                steps=steps_log,
                outcome=final_answer
            )
            print("\n💾 Session saved to memory.")
            return

        # If decision is "search", execute the search
        query = decision.get("query", "")
        if not query:
            print("   ⚠️ Empty query — stopping research")
            break

        # Check we haven't searched this before
        previous_queries = [f["query"] for f in findings]
        if query in previous_queries:
            print(f"   ⚠️ Duplicate query detected: '{query}' — stopping")
            break

        print(f"   🔍 Searching: '{query}'")

        result = run_tool({"action": "knowledge_base", "input": query})
        context = result.get("context", "No results found.")

        findings.append({"query": query, "result": context})
        steps_log.append({"decision": {"action": "knowledge_base", "input": query}, "result": result})

        print(f"   📄 Found: {context[:150]}...")
        # [:150] — print just the first 150 chars so terminal isn't flooded

# ----------------------
# THINK STEP — with retry logic
# ----------------------

def llm_think(state, past_context: str, max_retries: int = 3):
    base_prompt = f"""
You are an AI agent with memory of past sessions.

PAST SESSIONS (for reference only — do NOT repeat these actions, just learn from them):
{past_context}

The current goal may be different. Always focus on completing the CURRENT GOAL.

CURRENT GOAL:
{state["goal"]}

PREVIOUS STEPS THIS SESSION:
{state["history"]}

Decide what to do next.
Respond ONLY in valid JSON with NO extra text before or after:

{{"thought": "...", "action": "...", "input": "..."}}

Allowed actions and EXACTLY what to put in "input":
- weather_api  → input: city name only, e.g. "Abu Dhabi"
- calculator   → input: math expression ONLY, e.g. "25 * 4". NO words, just numbers and operators
- final_answer → input: a full sentence with ALL results. e.g. "Weather in Dubai is sunny at 34°C, and 25 * 10 = 250." NEVER leave this empty.
- knowledge_base → input: a search query describing what you want to know, e.g. "UAE visa requirements"


IMPORTANT RULES:
1. Use the right tool for the goal:
   - weather questions → weather_api, then final_answer
   - math → calculator, then final_answer
   - factual questions → knowledge_base, then final_answer
   Once you get a result from any tool, call final_answer IMMEDIATELY with the answer.
   Do NOT search the knowledge base more than once.
2. Never call the same action+input twice.
3. calculator input must be a pure math expression. Never a sentence.
4. final_answer must be a complete sentence. Never empty.
5. Do NOT invent extra steps after completing the goal.
"""

    for attempt in range(max_retries):

        if attempt == 0:
            prompt = base_prompt
        else:
            prompt = base_prompt + f"""

⚠️ PREVIOUS ATTEMPT {attempt} FAILED. Your response was not valid JSON or was missing required fields.
You MUST respond with ONLY this exact structure, nothing else:
{{"thought": "your reasoning", "action": "one of: weather_api, calculator, final_answer", "input": "correct input for that action"}}
"""
            print(f"  ⚠️ Retry attempt {attempt + 1}/{max_retries}")

        reply = call_llm(prompt)

        try:
            decision = json.loads(reply)
            if is_valid_decision(decision):
                return decision
            else:
                print(f"  ❌ Invalid decision structure: {decision}")

        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parse failed: {e}")
            print(f"  Raw reply was: {reply[:100]}")

    print("  💀 All retries exhausted. Using fallback.")
    return {
        "thought": "all retries failed",
        "action": "final_answer",
        "input": "Agent failed to produce a valid response after multiple attempts."
    }


# ----------------------
# TOOLS
# ----------------------

def run_tool(action):
    if action["action"] == "calculator":
        try:
            return {"result": eval(action["input"])}
        except:
            return {"error": "bad math"}

    if action["action"] == "weather_api":
        # BEFORE: return {"temperature": 34, "condition": "sunny"}
        # AFTER: real live API call
        result = get_weather(action["input"])
        return result

    if action["action"] == "knowledge_base":
        results = search(action["input"], DOCUMENTS, VECTORIZER, MATRIX)
        context = format_context(results)
        return {"context": context}

    return None

# ----------------------
# AGENT LOOP
# ----------------------

def run_agent(goal):

    # ── OBSERVABILITY: Start trace ─────────────────────────────
    trace = AgentTrace(goal)
    trace.log("agent_start", {"goal": goal})
    logger.info(f"Agent started — goal: {goal}")

    memory = load_memory()
    past_context = get_recent_context(memory)
    logger.info(f"Memory loaded. Past sessions: {len(memory['sessions'])}")

    # ── PHASE 1: PLAN ──────────────────────────────────────────
    logger.info("Planning steps...")
    print("\n🗺️  Planning steps...")

    plan = []
    for attempt in range(3):
        plan = llm_plan(goal, past_context)
        if is_valid_plan(plan):
            break
        logger.warning(f"Invalid plan on attempt {attempt + 1}")
        print(f"  ⚠️ Invalid plan on attempt {attempt + 1}, retrying...")

    if not plan:
        logger.error("Could not generate a valid plan. Aborting.")
        print("  💀 Could not generate a valid plan. Aborting.")
        trace.log("agent_end", {"outcome": "plan_failed"}, level="error")
        trace.save()
        print_metrics_summary()
        return

    trace.log("plan_generated", {"steps": len(plan), "plan": plan})
    print(f"\n📋 Plan ({len(plan)} steps):")
    for s in plan:
        print(f"   Step {s['step']}: {s['tool']} ← '{s['input']}'  ({s.get('reason', '')})")

    # ── PHASE 2: EXECUTE ───────────────────────────────────────
    logger.info("Executing plan...")
    print("\n⚙️  Executing plan...")

    steps_log = []
    results_summary = []

    for s in plan:
        print(f"\n--- Executing Step {s['step']}: {s['tool']} ---")
        logger.info(f"Executing step {s['step']}: {s['tool']} ← '{s['input']}'")
        trace.log("step_start", {"step": s["step"], "tool": s["tool"], "input": s["input"]})

        action = {"action": s["tool"], "input": s["input"]}

        # Tool execution with timing
        tool_start = time.time()
        result = run_tool(action)
        tool_duration = round((time.time() - tool_start) * 1000)

        success = result is not None and "error" not in result
        record_metric("tool_call", {
            "tool": s["tool"],
            "duration_ms": tool_duration,
            "success": success
        })

        trace.log("tool_result", {
            "tool": s["tool"],
            "duration_ms": tool_duration,
            "success": success,
            "result_preview": str(result)[:150]
        })

        if not success:
            logger.warning(f"Tool {s['tool']} returned error: {result}")
        else:
            logger.info(f"Tool {s['tool']} completed in {tool_duration}ms")

        print(f"   Result: {result}")

        step_record = {"decision": action, "result": result}
        steps_log.append(step_record)
        results_summary.append(f"Step {s['step']} ({s['tool']} → '{s['input']}'): {result}")

    # ── PHASE 3: SYNTHESIZE ────────────────────────────────────
    logger.info("Synthesizing final answer...")
    print("\n✍️  Synthesizing final answer...")

    synthesis_prompt = f"""
You are an AI agent. You executed a plan and collected results.

ORIGINAL GOAL:
{goal}

RESULTS FROM EACH STEP:
{chr(10).join(results_summary)}

Write a clear, complete final answer that addresses the original goal using ALL the results above.
Write in plain English. Be concise. Do not mention "steps" or "tools" — just answer the goal directly.
"""

    final_answer = call_llm(synthesis_prompt)
    print(f"\n✅ FINAL ANSWER:\n{final_answer}")

    logger.info(f"Final answer generated: {final_answer[:100]}...")
    trace.log("final_answer", {"answer": final_answer})

    log_session(
        memory=memory,
        goal=goal,
        steps=steps_log,
        outcome=final_answer
    )

    trace.log("agent_end", {"outcome": "success"})
    trace.save()
    print_metrics_summary()
    print("\n💾 Session saved to memory.")

if __name__ == "__main__":
    run_agent("Check weather in London and calculate 100 * 12")