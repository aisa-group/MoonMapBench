"""The log of Message Batches sent for one model and effort: results/<model>__<effort>/batches.jsonl.

One JSON object per line, appended in order while the results directory is locked:
- "submitting": written just before a batch is sent;
- "submitted" (batch_id): the API accepted it;
- "submit_rejected": the API refused it, so no batch exists;
- "collected" (batch_id): its answers are in responses.jsonl.
A "submitting" entry with no "submitted" or "submit_rejected" after it means that the process
died or the request failed while sending, so it is unknown whether a batch was created.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

BATCH_LOG = "batches.jsonl"


@dataclass
class BatchLog:
    pending: list[dict]  # "submitted" entries that are not collected yet, in order
    unresolved: dict | None  # a "submitting" entry without an outcome


def read_batch_log(path: Path) -> BatchLog:
    submitted: list[dict] = []
    collected: set[str] = set()
    unresolved = None
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                entry = json.loads(line)
                event = entry["event"]
                if event == "submitting":
                    if unresolved is not None:
                        raise RuntimeError(f"{path}:{line_no}: earlier submission has no outcome: {unresolved}")
                    unresolved = entry
                elif event in ("submitted", "submit_rejected"):
                    unresolved = None
                    if event == "submitted":
                        submitted.append(entry)
                elif event == "collected":
                    collected.add(entry["batch_id"])
                else:
                    raise RuntimeError(f"{path}:{line_no}: unknown event {entry}")
    return BatchLog([entry for entry in submitted if entry["batch_id"] not in collected], unresolved)


def append_batch_log(path: Path, entry: dict) -> None:
    entry["time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
