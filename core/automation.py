"""Local workflow recording and a cancellable, sequential workflow queue."""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from memory.config_manager import (
    get_tool_permission,
    save_tool_permission,
)

_NAME_RE = re.compile(r"^[\w][\w .-]{0,59}$", re.UNICODE)
_SENSITIVE_KEYS = {
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "authorization", "cookie",
}
_MAX_STEPS = 50
_MAX_VALUE_LENGTH = 2000


def _sanitize(value, depth: int = 0):
    if depth > 8:
        raise ValueError("Workflow parameters are nested too deeply.")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > _MAX_VALUE_LENGTH:
            raise ValueError("A workflow parameter is too long to save.")
        return value
    if isinstance(value, list):
        if len(value) > 100:
            raise ValueError("A workflow parameter contains too many items.")
        return [_sanitize(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 100:
            raise ValueError("A workflow parameter contains too many fields.")
        result = {}
        for key, item in value.items():
            name = str(key)
            normalized = name.lower().replace("-", "_")
            if any(part in normalized for part in _SENSITIVE_KEYS):
                raise ValueError(f"Sensitive parameter '{name}' cannot be recorded.")
            result[name] = _sanitize(item, depth + 1)
        return result
    raise ValueError(f"Unsupported workflow parameter type: {type(value).__name__}.")


class AutomationManager:
    """Stores reviewed workflows and runs them in a single background worker."""

    def __init__(self, path: Path, run_action: Callable[[str, dict], str],
                 logger: Callable[[str], None] = print):
        self._path = path
        self._run_action = run_action
        self._logger = logger
        self._lock = threading.RLock()
        self._recording: list[dict] | None = None
        self._queue: list[dict] = []
        self._tasks: list[dict] = []
        self._worker: threading.Thread | None = None
        self._active_task: dict | None = None
        self._cancel_active = threading.Event()

    def permission_items(self, tools: list[dict]) -> list[dict]:
        return [
            {"name": str(tool["name"]), "description": str(tool.get("description", "")),
             "allowed": get_tool_permission(str(tool["name"]))}
            for tool in tools
        ]

    def set_permission(self, name: str, allowed: bool) -> None:
        save_tool_permission(name, allowed)

    def allowed(self, name: str) -> bool:
        return get_tool_permission(name)

    def start_recording(self) -> None:
        with self._lock:
            if self._recording is not None:
                raise RuntimeError("A workflow is already being recorded.")
            self._recording = []

    def stop_recording(self) -> list[dict]:
        with self._lock:
            if self._recording is None:
                return []
            steps, self._recording = self._recording, None
            return json.loads(json.dumps(steps))

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording is not None

    def record_step(self, name: str, parameters: dict) -> None:
        if not self.allowed(name):
            return
        try:
            clean = _sanitize(parameters)
        except ValueError as exc:
            self._logger(f"[Workflows] Skipped recording {name}: {exc}")
            return
        with self._lock:
            if self._recording is None:
                return
            if len(self._recording) >= _MAX_STEPS:
                self._recording = None
                self._logger("[Workflows] Recording stopped at the 50-step limit.")
                return
            self._recording.append({"tool": name, "parameters": clean})

    def list_workflows(self) -> list[dict]:
        with self._lock:
            data = self._read()
            return [{"name": name, "steps": len(steps)}
                    for name, steps in sorted(data.items(), key=lambda item: item[0].lower())]

    def save_recording(self, name: str, steps: list[dict]) -> None:
        clean_name = str(name).strip()
        if not _NAME_RE.fullmatch(clean_name):
            raise ValueError("Use a workflow name of 1–60 letters, numbers, spaces, dots, _ or -.")
        if not steps:
            raise ValueError("The recording is empty.")
        if len(steps) > _MAX_STEPS:
            raise ValueError(f"Workflows are limited to {_MAX_STEPS} steps.")
        clean_steps = []
        for step in steps:
            if not isinstance(step, dict) or not isinstance(step.get("tool"), str):
                raise ValueError("A recorded step is invalid.")
            clean_steps.append({
                "tool": step["tool"],
                "parameters": _sanitize(step.get("parameters", {})),
            })
        with self._lock:
            data = self._read()
            data[clean_name] = clean_steps
            self._write(data)

    def delete_workflow(self, name: str) -> bool:
        with self._lock:
            data = self._read()
            if name not in data:
                return False
            del data[name]
            self._write(data)
            return True

    def enqueue(self, name: str) -> str:
        with self._lock:
            workflow = self._read().get(name)
            if workflow is None:
                raise ValueError(f"Workflow '{name}' was not found.")
            task = {
                "id": uuid.uuid4().hex[:8],
                "workflow": name,
                "status": "queued",
                "step": 0,
                "total": len(workflow),
                "detail": "Waiting to run",
                "at": time.time(),
            }
            self._tasks.append(task)
            self._queue.append({"task": task, "steps": workflow})
            self._ensure_worker()
            return task["id"]

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            for item in self._queue:
                task = item["task"]
                if task["id"] == task_id:
                    self._queue.remove(item)
                    task["status"] = "cancelled"
                    task["detail"] = "Cancelled before starting"
                    return True
            if (self._active_task is not None
                    and self._active_task["id"] == task_id
                    and self._active_task["status"] == "running"):
                self._cancel_active.set()
                return True
        return False

    def task_list(self) -> list[dict]:
        with self._lock:
            return [dict(task) for task in self._tasks[-20:]]

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._work, name="workflow-queue", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    return
                item = self._queue.pop(0)
                task = item["task"]
                self._active_task = task
                task["status"] = "running"
                task["detail"] = "Starting"
                self._cancel_active.clear()
            failed = False
            for index, step in enumerate(item["steps"], start=1):
                if self._cancel_active.is_set():
                    task["status"] = "cancelled"
                    task["detail"] = "Cancelled after the current step"
                    break
                name = step["tool"]
                if not self.allowed(name):
                    task["status"] = "blocked"
                    task["detail"] = f"Permission denied for {name}"
                    failed = True
                    break
                task["step"] = index
                task["detail"] = f"Running {name}"
                try:
                    result = self._run_action(name, step["parameters"])
                except Exception as exc:
                    result = f"Action failed: {exc}"
                result = str(result or "")
                task["detail"] = result[:180] or f"Completed {name}"
                if any(marker in result.lower() for marker in (
                        "failed", "not available", "permission denied",
                        "confirmation_pending", "unknown tool", "unknown action")):
                    task["status"] = "blocked"
                    failed = True
                    break
            if task["status"] == "running":
                if self._cancel_active.is_set():
                    task["status"] = "cancelled"
                    task["detail"] = "Cancelled after the current step"
                else:
                    task["status"] = "failed" if failed else "completed"
                    task["detail"] = ("Workflow stopped after an unsuccessful step"
                                      if failed else "Workflow completed")
            self._logger(
                f"[Workflows] {task['workflow']} — {task['status']}: {task['detail']}")
            with self._lock:
                self._active_task = None

    def _read(self) -> dict[str, list[dict]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            self._logger(f"[Workflows] Could not read {self._path}: {exc}")
            raise ValueError(f"Could not read saved workflows: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Workflow file must contain an object.")
        workflows = {}
        for name, steps in data.items():
            if (not isinstance(name, str) or not _NAME_RE.fullmatch(name)
                    or not isinstance(steps, list) or not steps
                    or len(steps) > _MAX_STEPS):
                raise ValueError("Saved workflow data has an invalid name or step list.")
            clean_steps = []
            for step in steps:
                if (not isinstance(step, dict)
                        or not isinstance(step.get("tool"), str)
                        or not isinstance(step.get("parameters", {}), dict)):
                    raise ValueError(f"Workflow '{name}' contains an invalid step.")
                clean_steps.append({
                    "tool": step["tool"],
                    "parameters": _sanitize(step.get("parameters", {})),
                })
            workflows[name] = clean_steps
        return workflows

    def _write(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        temporary.replace(self._path)
