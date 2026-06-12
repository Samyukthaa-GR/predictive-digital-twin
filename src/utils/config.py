from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text)
    except ModuleNotFoundError:
        loaded = _load_simple_yaml(text)

    if not isinstance(loaded, dict):
        raise ValueError(f"Config must parse to a dictionary: {path}")
    return loaded


def _load_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]

        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("Invalid simple YAML: list item without list parent")
            parent.append(_coerce_scalar(line[2:].strip()))
            continue

        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid simple YAML line: {raw_line}")

        key = key.strip()
        value = value.strip()

        if value:
            parent[key] = _coerce_scalar(value)
            continue

        next_container: list[Any] | dict[str, Any]
        next_container = [] if _next_significant_line_is_list(lines, index) else {}
        parent[key] = next_container
        stack.append((indent, next_container))

    return root


def _next_significant_line_is_list(lines: list[str], current_index: int) -> bool:
    current_line = lines[current_index]
    current_indent = len(current_line) - len(current_line.lstrip(" "))

    for line in lines[current_index + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        return indent > current_indent and line.strip().startswith("- ")
    return False


def _coerce_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
