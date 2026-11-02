"""Part C — Task State & Dependency Tracker (§5.4 of Review Plan v3).

Maintains lightweight sequential execution state for multi-action user requests:
- Tracks tasks across states: 'pending', 'blocked', 'done'.
- Enforces dependencies (e.g. read before write, Step 1 before Step 2).
- Distinguishes legitimate repeated actions from model looping.
- Prevents premature termination when steps are still pending.

Usage:
    tracker = TaskTracker(request_text, installed_tools)
    next_task = tracker.get_next_actionable_task()
    tracker.mark_done(task_id, tool_name, result)
"""
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
    """Manages sequential execution and dependency unblocking."""

    def __init__(self, raw_transcript: str = ""):
        self.raw_transcript = raw_transcript
        self.tasks: list[SubTask] = []
        self.completed_task_ids: set[str] = set()

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

    def mark_executed(self, task_id: str, tool_name: str, result: dict):
        """Record an execution step and update task status."""
        for t in self.tasks:
            if t.task_id == task_id:
                t.executed_tools.append(tool_name)
                # If write succeeded or read retrieved necessary data, mark done
                if result.get("ok", True):
                    t.status = "done"
                    t.completion_evidence = result
                    self.completed_task_ids.add(task_id)
                break

    def is_all_completed(self) -> bool:
        """Check if all planned tasks have completed."""
        return all(t.status == "done" for t in self.tasks)

    def has_repeating_loop(self, tool_name: str, arguments: dict, history_calls: list[dict]) -> bool:
        """Detect degenerate looping (calling identical tool with identical args > 2 times)."""
        recent = history_calls[-3:]
        count = sum(1 for c in recent if c.get("name") == tool_name and c.get("arguments") == arguments)
        return count >= 2
