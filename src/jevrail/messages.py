from __future__ import annotations

import copy
from typing import Any


def _is_pair(message: Any) -> bool:
    return isinstance(message, tuple) and len(message) == 2 and isinstance(message[0], str)


def message_text(message: Any) -> str:
    if _is_pair(message):
        content = message[1]
    elif isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def message_role(message: Any) -> str:
    if _is_pair(message):
        raw = message[0]
    elif isinstance(message, dict):
        raw = message.get("role") or message.get("type") or ""
    else:
        raw = getattr(message, "role", None) or getattr(message, "type", None) or ""
        class_name = type(message).__name__.lower()
        if "human" in class_name:
            return "user"
        if class_name.startswith("ai") or "assistant" in class_name:
            return "assistant"
        if "tool" in class_name:
            return "tool"
    role = str(raw).lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"ai", "assistant"}:
        return "assistant"
    return role


def latest_text(messages: list[Any], roles: set[str]) -> tuple[int, Any, str] | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message_role(message) in roles:
            return index, message, message_text(message)
    return None


def with_content(message: Any, content: str) -> Any:
    if _is_pair(message):
        return (message[0], content)
    if isinstance(message, dict):
        copied = dict(message)
        copied["content"] = content
        return copied
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    cloned = copy.copy(message)
    cloned.content = content
    return cloned
