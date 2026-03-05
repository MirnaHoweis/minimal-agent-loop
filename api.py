from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import time
from datetime import datetime
from agent import run_agent, load_memory

# ----------------------
# Why import from agent.py instead of rewriting?
# Your agent logic is already built and tested.
# The API's only job is to receive requests and call that logic.
# This separation means you can update agent logic without touching API code.
# ----------------------

app = FastAPI(
    title="Agentic AI API",
    description="API wrapper for autonomous agent system",
    version="1.0.0"
)
# FastAPI automatically generates documentation from these fields.
# Visit http://localhost:8000/docs after starting the server.

# ADD THIS BLOCK right after app is created
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REQUEST AND RESPONSE MODELS ───────────────────────────────────────

class AgentRequest(BaseModel):
    """
    Defines what the caller must send.
    Pydantic automatically validates the incoming JSON against this model.
    If a required field is missing, FastAPI returns a 422 error automatically.
    """
    goal: str
    # The goal is the only required field — the agent handles everything else.

    class Config:
        json_schema_extra = {
            "example": {
                "goal": "Check the weather in Dubai and calculate 25 * 4"
            }
        }


class AgentResponse(BaseModel):
    """
    Defines what the API returns.
    Consistent response structure = easier for callers to parse.
    """
    request_id: str        # unique ID for this request — useful for debugging
    goal: str              # echo back what was requested
    status: str            # "success" or "error"
    answer: str            # the agent's final answer
    duration_seconds: float  # how long it took
    timestamp: str         # when it ran


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# ── JOB STORE ─────────────────────────────────────────────────────────

# For async jobs we need somewhere to store results temporarily.
# In production this would be Redis or a database.
# For learning purposes, a plain dict works fine.
jobs = {}


# ── ENDPOINTS ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Standard health check endpoint.
    Every production API has this — monitoring systems ping it to confirm
    the service is alive. Returns 200 OK if the server is running.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/run", response_model=AgentResponse)
def run_agent_sync(request: AgentRequest):
    """
    Synchronous endpoint — waits for the agent to finish, then responds.
    Simple but blocks the caller until the agent completes.
    Good for quick tasks. For long-running tasks use /run/async below.
    """
    request_id = str(uuid.uuid4())[:8]
    # uuid4() generates a random unique ID.
    # [:8] takes just the first 8 characters — short enough to read, unique enough to track.

    start_time = time.time()

    try:
        # Capture the agent's output.
        # run_agent() currently prints to terminal — we need to capture its outcome.
        # We do this by reading memory before and after the run.
        memory_before = load_memory()
        sessions_before = len(memory_before["sessions"])

        run_agent(request.goal)

        memory_after = load_memory()
        sessions_after = memory_after["sessions"]

        # The agent saved a new session — read its outcome as the answer
        if len(sessions_after) > sessions_before:
            answer = sessions_after[-1]["outcome"]
        else:
            answer = "Agent completed but no outcome was recorded."

        duration = round(time.time() - start_time, 2)

        return AgentResponse(
            request_id=request_id,
            goal=request.goal,
            status="success",
            answer=answer,
            duration_seconds=duration,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent failed: {str(e)}"
        )
        # HTTPException is FastAPI's way of returning error responses.
        # status_code=500 means "internal server error".
        # The caller receives a proper JSON error, not a Python traceback.


@app.post("/run/async")
def run_agent_async(request: AgentRequest, background_tasks: BackgroundTasks):
    """
    Asynchronous endpoint — returns immediately with a job ID,
    runs the agent in the background.
    
    Why is this useful?
    Agents can take 30-60 seconds. A synchronous call would leave the
    caller waiting with no feedback. With async, the caller gets a job ID
    immediately, then polls /status/{job_id} to check progress.
    This is the standard pattern for long-running AI tasks.
    """
    job_id = str(uuid.uuid4())[:8]

    # Register the job as "running" immediately
    jobs[job_id] = {
        "status": "running",
        "goal": request.goal,
        "started_at": datetime.now().isoformat(),
        "answer": None
    }

    def execute_agent(job_id: str, goal: str):
        """This function runs in the background while the API responds immediately."""
        try:
            memory_before = load_memory()
            sessions_before = len(memory_before["sessions"])

            run_agent(goal)

            memory_after = load_memory()
            sessions_after = memory_after["sessions"]

            if len(sessions_after) > sessions_before:
                answer = sessions_after[-1]["outcome"]
            else:
                answer = "Completed — no outcome recorded."

            jobs[job_id]["status"] = "complete"
            jobs[job_id]["answer"] = answer
            jobs[job_id]["completed_at"] = datetime.now().isoformat()

        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)

    # Schedule the agent to run in the background
    background_tasks.add_task(execute_agent, job_id, request.goal)

    # Return immediately — caller doesn't wait for the agent
    return {
        "job_id": job_id,
        "status": "running",
        "message": f"Agent started. Poll /status/{job_id} for results.",
        "poll_url": f"/status/{job_id}"
    }


@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """
    Check the status of an async job.
    The caller polls this endpoint until status is "complete" or "failed".
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return jobs[job_id]


@app.get("/history")
def get_history(limit: int = 5):
    """
    Returns recent agent sessions from memory.
    Useful for building a frontend that shows past runs.
    limit parameter lets callers control how many results they want.
    """
    memory = load_memory()
    sessions = memory["sessions"]
    recent = sessions[-limit:][::-1]
    # [-limit:] = last N sessions
    # [::-1] = reverse so newest is first

    return {
        "total_sessions": len(sessions),
        "showing": len(recent),
        "sessions": [
            {
                "timestamp": s["timestamp"],
                "goal": s["goal"],
                "outcome": s["outcome"][:200],
                # Truncate long outcomes for the list view
                "steps_taken": len(s["steps_taken"])
            }
            for s in recent
        ]
    }