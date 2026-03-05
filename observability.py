import json
import time
import logging
import os
from datetime import datetime
from functools import wraps

# ----------------------
# Why a separate file?
# Observability is infrastructure — it supports your agent but isn't
# part of its logic. Keeping it separate means you can add it to
# any agent without changing agent code, just importing it.
# ----------------------


# ── LEVEL 1: STRUCTURED LOGGING ───────────────────────────────────────

LOG_FILE = "agent_logs.jsonl"
# .jsonl = JSON Lines format — each line is a complete JSON object.
# This is the standard format for production logs because:
# - each line is independently parseable
# - you can append without loading the whole file
# - log analysis tools (Splunk, Datadog, etc.) all understand it

def setup_logger(name: str) -> logging.Logger:
    """
    Creates a logger that writes to both terminal AND a file.
    This means you still see output while developing,
    but every message is also permanently saved.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # DEBUG is the lowest level — captures everything.
    # In production you'd set this to INFO to reduce noise.

    # Prevent adding duplicate handlers if setup_logger is called twice
    if logger.handlers:
        return logger

    # Handler 1: write to terminal (what you see now)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # INFO and above go to terminal — DEBUG messages stay file-only
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)

    # Handler 2: write to file (permanent record)
    file_handler = logging.FileHandler("agent_run.log")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ── LEVEL 2: METRICS ──────────────────────────────────────────────────

METRICS_FILE = "agent_metrics.jsonl"

def record_metric(event: str, data: dict):
    """
    Writes a single metric event to the metrics file.
    
    Every important event gets recorded:
    - agent_start, agent_end
    - tool_call, tool_success, tool_failure
    - llm_call, llm_retry, llm_failure
    - plan_generated, research_complete
    
    Over time these records let you answer questions like:
    "What percentage of calculator calls succeed?"
    "How many retries does the LLM need on average?"
    """
    record = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        **data
        # ** unpacks the data dict into the record dict
        # So {"event": "tool_call", **{"tool": "calculator"}}
        # becomes {"event": "tool_call", "tool": "calculator"}
    }
    with open(METRICS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
        # "a" = append mode — never overwrites existing records
        # Each metric is one line of JSON


def load_metrics() -> list:
    """Reads all metrics for analysis."""
    if not os.path.exists(METRICS_FILE):
        return []
    records = []
    with open(METRICS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_metrics_summary():
    """
    Reads all recorded metrics and prints a performance summary.
    Call this anytime to see how your agent has been performing.
    """
    records = load_metrics()
    if not records:
        print("No metrics recorded yet.")
        return

    # Count events by type
    event_counts = {}
    for r in records:
        event = r["event"]
        event_counts[event] = event_counts.get(event, 0) + 1

    # Calculate average LLM response time
    llm_times = [
        r["duration_ms"] for r in records
        if r["event"] == "llm_call" and "duration_ms" in r
    ]
    avg_llm_time = sum(llm_times) / len(llm_times) if llm_times else 0

    # Calculate tool success rate
    tool_calls = event_counts.get("tool_call", 0)
    tool_failures = event_counts.get("tool_failure", 0)
    success_rate = ((tool_calls - tool_failures) / tool_calls * 100) if tool_calls > 0 else 0

    # Calculate retry rate
    total_llm = event_counts.get("llm_call", 0)
    retries = event_counts.get("llm_retry", 0)
    retry_rate = (retries / total_llm * 100) if total_llm > 0 else 0

    print("\n" + "="*50)
    print("📊 AGENT PERFORMANCE METRICS")
    print("="*50)
    print(f"Total agent runs:       {event_counts.get('agent_start', 0)}")
    print(f"Successful completions: {event_counts.get('agent_end', 0)}")
    print(f"Total LLM calls:        {total_llm}")
    print(f"LLM retry rate:         {retry_rate:.1f}%")
    print(f"Avg LLM response time:  {avg_llm_time:.0f}ms")
    print(f"Total tool calls:       {tool_calls}")
    print(f"Tool success rate:      {success_rate:.1f}%")
    print("="*50)

    # Break down by tool
    tool_records = [r for r in records if r["event"] == "tool_call"]
    if tool_records:
        print("\nTool usage breakdown:")
        tool_usage = {}
        for r in tool_records:
            tool = r.get("tool", "unknown")
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
        for tool, count in sorted(tool_usage.items()):
            print(f"   {tool}: {count} calls")
    print()


# ── LEVEL 3: TRACING ──────────────────────────────────────────────────

class AgentTrace:
    """
    Records a complete trace of one agent run.
    
    A trace is a timeline — every event that happened during one run,
    in order, with timestamps and durations.
    
    Think of it like a flight recorder for your agent.
    If something goes wrong, you open the trace and see exactly what happened.
    """

    def __init__(self, goal: str):
        self.goal = goal
        self.trace_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Simple ID based on timestamp — unique enough for local use
        self.start_time = time.time()
        self.events = []
        self.logger = setup_logger(f"trace.{self.trace_id}")

    def log(self, event_type: str, data: dict, level: str = "info"):
        """Add one event to the trace."""
        elapsed = round((time.time() - self.start_time) * 1000)
        # elapsed = milliseconds since this trace started
        # Relative time is more useful than absolute time for debugging

        event = {
            "elapsed_ms": elapsed,
            "event": event_type,
            **data
        }
        self.events.append(event)

        # Also write to the structured log
        message = f"[{elapsed}ms] {event_type} — {data}"
        getattr(self.logger, level)(message)
        # getattr(logger, "info") is the same as logger.info
        # This lets us choose the log level dynamically

        # Also record as a metric
        record_metric(event_type, {"trace_id": self.trace_id, **data})

    def save(self):
        """Save the complete trace to a file."""
        trace_data = {
            "trace_id": self.trace_id,
            "goal": self.goal,
            "total_duration_ms": round((time.time() - self.start_time) * 1000),
            "event_count": len(self.events),
            "events": self.events
        }

        os.makedirs("traces", exist_ok=True)
        # Create a traces/ folder if it doesn't exist
        # exist_ok=True means don't crash if it already exists

        filepath = f"traces/trace_{self.trace_id}.json"
        with open(filepath, "w") as f:
            json.dump(trace_data, f, indent=2)

        self.logger.info(f"Trace saved → {filepath}")
        return filepath