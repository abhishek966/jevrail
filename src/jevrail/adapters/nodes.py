from __future__ import annotations

from typing import Any, Callable

from jevrail.engine import GuardEngine, GuardrailBlocked
from jevrail.messages import latest_text, with_content
from jevrail.policy import GuardPolicy


class _Goto:
    def __init__(self, update: dict[str, Any], goto: str) -> None:
        self.update = update
        self.goto = goto


def _command(update: dict[str, Any]) -> Any:
    try:
        from langgraph.graph import END
        from langgraph.types import Command

        return Command(update=update, goto=END)
    except ImportError:
        return _Goto(update, "__end__")


def _assistant(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content}


def input_guard(
    policy: GuardPolicy | None = None,
    engine: GuardEngine | None = None,
    classifier: object | None = None,
) -> Callable[[dict[str, Any]], Any]:
    """A graph node that checks the latest user or tool message."""
    active = policy or GuardPolicy.default()
    active_engine = engine or GuardEngine(active, classifier)

    def node(state: dict[str, Any]) -> Any:
        messages = list(state.get("messages") or [])
        found = latest_text(messages, {"user", "tool"})
        if found is None:
            return {}
        decision = active_engine.judge("input", found[2])
        if decision.action == "block":
            if active.on_block == "raise":
                raise GuardrailBlocked(decision)
            return _command({"messages": [_assistant(decision.refusal or active.refusal_message)]})
        if decision.action == "redact":
            return {"messages": [with_content(found[1], decision.text)]}
        return {}

    return node


def output_guard(
    policy: GuardPolicy | None = None,
    engine: GuardEngine | None = None,
    classifier: object | None = None,
) -> Callable[[dict[str, Any]], Any]:
    """A graph node that checks the latest assistant message."""
    active = policy or GuardPolicy.default()
    active_engine = engine or GuardEngine(active, classifier)

    def node(state: dict[str, Any]) -> Any:
        messages = list(state.get("messages") or [])
        found = latest_text(messages, {"assistant"})
        if found is None:
            return {}
        decision = active_engine.judge("output", found[2])
        if decision.action == "block":
            if active.on_block == "raise":
                raise GuardrailBlocked(decision)
            return _command({"messages": [_assistant(decision.refusal or active.refusal_message)]})
        if decision.action == "redact":
            return {"messages": [with_content(found[1], decision.text)]}
        return {}

    return node
