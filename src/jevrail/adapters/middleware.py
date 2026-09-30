from __future__ import annotations

import json
from typing import Any

from jevrail.engine import GuardEngine, GuardrailBlocked
from jevrail.messages import latest_text, message_text, with_content
from jevrail.policy import GuardPolicy

try:
    from langchain.agents.middleware import AgentMiddleware, hook_config
except ImportError:  # pragma: no cover - exercised when langchain is absent

    class AgentMiddleware:  # type: ignore[no-redef]
        pass

    def hook_config(**_kwargs: Any):  # type: ignore[no-redef]
        def decorate(fn: Any) -> Any:
            return fn

        return decorate


class _Message:
    def __init__(self, role: str, content: str, message_id: str | None = None) -> None:
        self.role = role
        self.type = role
        self.content = content
        self.id = message_id or "jevrail"


def _ai(content: str) -> Any:
    try:
        from langchain.messages import AIMessage

        return AIMessage(content=content)
    except ImportError:
        return _Message("assistant", content)


def _tool_message(content: str, tool_call_id: str | None) -> Any:
    try:
        from langchain.messages import ToolMessage

        return ToolMessage(content=content, tool_call_id=tool_call_id or "jevrail")
    except ImportError:
        message = _Message("tool", content, tool_call_id or "jevrail")
        message.tool_call_id = tool_call_id or "jevrail"
        return message


def _model_response(messages: list[Any]) -> Any:
    try:
        from langchain.agents.middleware import ModelResponse

        return ModelResponse(result=messages)
    except ImportError:
        return type("ModelResponse", (), {"result": messages})()


class JevRail(AgentMiddleware):
    """LangChain ``create_agent`` middleware.

    ``before_model`` blocks unsafe user or tool text. ``after_model`` blocks
    unsafe model text. ``wrap_model_call`` redacts input before the model sees
    it. ``wrap_tool_call`` covers tool calls and tool results when those stages
    are listed on the policy.
    """

    def __init__(
        self,
        policy: GuardPolicy | None = None,
        engine: GuardEngine | None = None,
        classifier: object | None = None,
    ) -> None:
        super().__init__()
        self.policy = policy or GuardPolicy.default()
        self.engine = engine or GuardEngine(self.policy, classifier)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        found = latest_text(list(state.get("messages") or []), {"user", "tool"})
        if found is None:
            return None
        decision = self.engine.judge("input", found[2])
        if decision.action != "block":
            return None
        self._raise_if_needed(decision)
        return {"messages": [_ai(decision.refusal or self.policy.refusal_message)], "jump_to": "end"}

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        found = latest_text(list(state.get("messages") or []), {"assistant"})
        if found is None:
            return None
        decision = self.engine.judge("output", found[2])
        if decision.action == "block":
            self._raise_if_needed(decision)
            return {
                "messages": [_ai(decision.refusal or self.policy.refusal_message)],
                "jump_to": "end",
            }
        if decision.action == "redact":
            return {"messages": [with_content(found[1], decision.text)]}
        return None

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        messages = list(request.messages)
        found = latest_text(messages, {"user", "tool"})
        if found is not None:
            index, message, text = found
            decision = self.engine.judge("input", text)
            if decision.action == "block":
                self._raise_if_needed(decision)
                return _model_response([_ai(decision.refusal or self.policy.refusal_message)])
            if decision.action == "redact":
                messages[index] = with_content(message, decision.text)
                request = request.override(messages=messages)
        return handler(request)

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        name, args, call_id = _tool_parts(request)
        payload = json.dumps({"name": name, "args": args}, default=str)
        decision = self.engine.judge("tool_call", payload, tool_name=name)
        if decision.action == "block":
            self._raise_if_needed(decision)
            return _tool_message(decision.refusal or self.policy.refusal_message, call_id)
        result = handler(request)
        outgoing = self.engine.judge("tool_result", message_text(result))
        if outgoing.action == "block":
            self._raise_if_needed(outgoing)
            return _tool_message(outgoing.refusal or self.policy.refusal_message, call_id)
        if outgoing.action == "redact":
            return with_content(result, outgoing.text)
        return result

    async def abefore_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    async def aafter_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        messages = list(request.messages)
        found = latest_text(messages, {"user", "tool"})
        if found is not None:
            index, message, text = found
            decision = self.engine.judge("input", text)
            if decision.action == "block":
                self._raise_if_needed(decision)
                return _model_response([_ai(decision.refusal or self.policy.refusal_message)])
            if decision.action == "redact":
                messages[index] = with_content(message, decision.text)
                request = request.override(messages=messages)
        result = handler(request)
        if hasattr(result, "__await__"):
            result = await result
        return result

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        name, args, call_id = _tool_parts(request)
        payload = json.dumps({"name": name, "args": args}, default=str)
        decision = self.engine.judge("tool_call", payload, tool_name=name)
        if decision.action == "block":
            self._raise_if_needed(decision)
            return _tool_message(decision.refusal or self.policy.refusal_message, call_id)
        result = handler(request)
        if hasattr(result, "__await__"):
            result = await result
        outgoing = self.engine.judge("tool_result", message_text(result))
        if outgoing.action == "block":
            self._raise_if_needed(outgoing)
            return _tool_message(outgoing.refusal or self.policy.refusal_message, call_id)
        if outgoing.action == "redact":
            return with_content(result, outgoing.text)
        return result

    def _raise_if_needed(self, decision: Any) -> None:
        if self.policy.on_block == "raise":
            raise GuardrailBlocked(decision)


def _tool_parts(request: Any) -> tuple[str | None, Any, str | None]:
    call = getattr(request, "tool_call", None)
    if call is None and isinstance(request, dict):
        call = request.get("tool_call")
    if isinstance(call, dict):
        return call.get("name"), call.get("args") or {}, call.get("id")
    if call is None:
        return None, {}, None
    return getattr(call, "name", None), getattr(call, "args", {}) or {}, getattr(call, "id", None)
