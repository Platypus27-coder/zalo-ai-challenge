"""Part C — Task State & Dependency Tracker (§8.3 of Plan C).

Maintains sequential execution state for multi-action user requests:
- Tracks subgoals and completion status based on concrete observation evidence.
- Ensures read-for-write steps do not prematurely terminate write goals.
- Fingerprints calls to prevent degenerate infinite looping without blocking legitimate repeated actions.
"""
from __future__ import annotations

import json
from typing import Any, Optional


class SubTask:
    """A sub-task within a multi-action user request."""

    def __init__(
        self,
        task_id: str,
        description: str,
        app: str,
        required_tool_kind: str = "write",
        depends_on: Optional[list[str]] = None
    ):
        self.task_id = task_id
        self.description = description
        self.app = app
        self.required_tool_kind = required_tool_kind  # 'read' or 'write'
        self.depends_on = depends_on or []
        self.status = "blocked" if self.depends_on else "pending"
        self.executed_tools: list[str] = []
        self.completion_evidence: Optional[dict] = None

    def can_execute(self, completed_tasks: set[str]) -> bool:
        return self.status == "pending" or all(dep in completed_tasks for dep in self.depends_on)


class TaskTracker:
    """Manages sequential execution, dependency tracking, and loop prevention."""

    def __init__(self, raw_transcript: str = ""):
        self.raw_transcript = raw_transcript
        self.tasks: list[SubTask] = []
        self.completed_task_ids: set[str] = set()
        self.executed_calls: list[tuple[str, str, str]] = []  # [(tool_name, canonical_args_json, tool_kind)]
        self.has_write_happened = False
        self.has_ask_happened = False
        self.write_count = 0
        self.seen_reads: dict[tuple[str, str], int] = {}

    def add_task(
        self,
        task_id: str,
        description: str,
        app: str,
        required_tool_kind: str = "write",
        depends_on: Optional[list[str]] = None
    ) -> SubTask:
        task = SubTask(task_id, description, app, required_tool_kind, depends_on)
        self.tasks.append(task)
        return task

    def get_actionable_tasks(self) -> list[SubTask]:
        """Return all tasks that are currently unblocked and pending."""
        actionable = []
        for t in self.tasks:
            if t.status in ("pending", "blocked") and t.can_execute(self.completed_task_ids):
                t.status = "pending"
                actionable.append(t)
        return actionable

    def record_call(self, tool_name: str, arguments: dict, result: dict, tool_kind: str = "write"):
        """Record an executed tool call and update internal tracking states."""
        canon_args = json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False)
        self.executed_calls.append((tool_name, canon_args, tool_kind))

        if tool_name == "sys.ask_missing_info":
            self.has_ask_happened = True
        elif tool_kind == "write":
            self.has_write_happened = True
            self.write_count += 1
        elif tool_kind == "read":
            self.seen_reads[(tool_name, canon_args)] = self.write_count

    def mark_executed(self, task_id: str, tool_name: str, result: dict):
        """Record an execution step and update task status."""
        for t in self.tasks:
            if t.task_id == task_id:
                t.executed_tools.append(tool_name)
                # Successful execution marks done
                if result.get("ok", True):
                    t.status = "done"
                    t.completion_evidence = result
                    self.completed_task_ids.add(task_id)
                break

    def is_all_completed(self) -> bool:
        """Check if all planned tasks have completed."""
        if not self.tasks:
            return True
        return all(t.status == "done" for t in self.tasks)

    def is_looping(self, tool_name: str, arguments: dict, tool_kind: str = "write") -> bool:
        """Detect degenerate looping:
        1. Calling the exact same WRITE tool with exact same arguments even once before -> loop (prevent duplicate writes)!
        2. Calling the exact same READ tool with exact same arguments >= 2 times -> loop!
        3. Alternating cycles (e.g. A -> B -> A -> B) -> loop!
        """
        canon_args = json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False)
        fingerprint = (tool_name, canon_args)
        
        # 1. Duplicate write call
        if tool_kind == "write":
            count = sum(1 for (name, args_str, kind) in self.executed_calls if (name, args_str) == fingerprint)
            if count >= 1:
                return True
        else:
            # Read tool: max 2 identical calls
            count = sum(1 for (name, args_str, kind) in self.executed_calls if (name, args_str) == fingerprint)
            if count >= 2:
                return True

        # 2. Cycle detection: check if last calls form an alternating cycle of period 2 or 3
        # e.g. [A, B, A, B] or [A, B, C, A, B, C]
        history = [call[0] for call in self.executed_calls] + [tool_name]
        if len(history) >= 4 and history[-1] == history[-3] and history[-2] == history[-4]:
            return True
        if len(history) >= 6 and history[-1] == history[-4] and history[-2] == history[-5] and history[-3] == history[-6]:
            return True

        return False
