# Minimal Agent Loop

A production-style autonomous AI agent system built from scratch in Python.
Demonstrates core agentic AI engineering concepts used in real AI products.

---

## What This Is

This project implements a complete agentic AI system where an LLM autonomously:
- Plans multi-step strategies to achieve goals
- Executes tools (weather API, calculator, knowledge base search)
- Remembers past sessions across runs
- Retries failed operations with error-aware prompting
- Logs every decision with full observability
- Serves its capabilities over a REST API with a browser frontend

Built as part of a 10-week Agentic AI Engineer training program targeting
production AI roles.

---

## Architecture
```
Browser UI (frontend.html)
        ↓
FastAPI REST API (api.py)
        ↓
Agent Core (agent.py)
    ├── Planner          — LLM decomposes goal into ordered steps
    ├── Executor         — runs each step with retry logic
    ├── Tool Registry    — calculator, weather API, knowledge base
    ├── Memory System    — persistent JSON store across sessions
    └── Observability    — structured logs, metrics, traces
        ↓
Local LLM via Ollama (llama3)
```

---

## Key Features

**Plan-and-Execute Architecture**
The agent separates planning from execution. It first generates a complete
step-by-step plan, then executes each step in order — matching the pattern
used by AutoGPT, LangChain Plan-and-Execute, and production AI assistants.

**Persistent Memory**
Every session is saved to disk. The agent reads past sessions at startup
and injects relevant history into its prompts — giving a stateless LLM
the appearance of long-term memory.

**Retrieval-Augmented Generation (RAG)**
TF-IDF based semantic search over a local knowledge base. The agent
retrieves relevant document chunks and grounds its answers in real content
instead of hallucinating.

**Retry Logic with Output Validation**
Every LLM call is validated against a strict schema. Invalid responses
trigger error-aware retries with progressively stricter prompts — up to 3
attempts before graceful fallback.

**Real API Integration**
Live weather data via Open-Meteo API with geocoding via Geopy.
No mock data — real coordinates, real forecasts.

**Full Observability Stack**
- Structured logs → `agent_run.log`
- Metrics (latency, success rate, retry rate) → `agent_metrics.jsonl`
- Per-run traces with millisecond timing → `traces/`

**REST API + Frontend**
FastAPI wrapper with sync and async endpoints. Browser UI with
real-time polling, session history, and dark mode interface.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM Runtime | Ollama (llama3) |
| Agent Framework | Built from scratch |
| API | FastAPI + Uvicorn |
| RAG Search | scikit-learn TF-IDF |
| Geocoding | Geopy + Nominatim |
| Weather Data | Open-Meteo API |
| Memory | JSON file store |
| Observability | Python logging + custom metrics |

---

## Getting Started

**Prerequisites**
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Pull the LLM**
```bash
ollama pull llama3
```

**Run the agent directly**
```bash
python agent.py
```

**Start the API server**
```bash
python -m uvicorn api:app --reload
```

**Open the frontend**

Open `frontend.html` in your browser.

---

## Example Goals
```
"Check the weather in Dubai and calculate 25 * 4"
"What are the visa requirements for visiting the UAE?"
"Research UAE business setup requirements and calculate total cost 
 if registration is 15000 AED and office rent is 8000 AED per month"
```

---

## What I Learned Building This

**Prompt engineering is debugging.**
Every agent failure was a prompt failure — vague instructions produce
unpredictable behavior. The fix is always specificity: concrete examples,
explicit rules, and format constraints.

**Code enforces, prompts guide.**
Soft behaviors (write a complete sentence) can be handled by prompts.
Hard constraints (never call a tool more than once) must be enforced in code.
Learning to distinguish between the two is core to production agent engineering.

**Observability is not optional.**
Without structured logs and traces, debugging agent behavior is guesswork.
Every production AI system needs the ability to reconstruct exactly what
happened during any given run.

**Separation of concerns scales.**
Building tools as separate modules (weather_tool.py, rag_engine.py,
observability.py) meant each component could be tested, swapped, and
extended without touching agent logic. This is why real agent frameworks
use plugin architectures.

---

## Roadmap

- [ ] Replace TF-IDF with vector embeddings (sentence-transformers)
- [ ] Swap JSON memory for SQLite
- [ ] Add streaming responses to the API
- [ ] Multi-agent orchestration (planner agent + executor agents)
- [ ] Docker containerization for deployment