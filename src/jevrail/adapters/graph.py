from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Any

from jevrail.engine import GuardEngine, GuardrailBlocked
from jevrail.messages import latest_text, message_role, message_text, with_content
from jevrail.policy import GuardPolicy

STREAM_MODES = frozenset({"values", "updates"})


@dataclass(frozen=True)
class _StreamLayout:
    """How LangGraph shapes stream chunks for one ``stream`` call."""

    modes: tuple[str, ...]
    multi: bool
    subgraphs: bool

    def unpack(self, chunk: Any) -> tuple[tuple, str, Any]:
        if self.subgraphs and self.multi:
            namespace, mode, payload = chunk
            return tuple(namespace), mode, payload
        if self.subgraphs:
            namespace, payload = chunk
            return tuple(namespace), self.modes[0], payload
        if self.multi:
            mode, payload = chunk
            return (), mode, payload
        return (), self.modes[0], chunk

    def pack(self, namespace: tuple, mode: str, payload: Any) -> Any:
        if self.subgraphs and self.multi:
            return (namespace, mode, payload)
        if self.subgraphs:
            return (namespace, payload)
        if self.multi:
            return (mode, payload)
        return payload

    def refusal(self, text: str, node: str) -> list[Any]:
        message = {"role": "assistant", "content": text}
        chunks = []
        for mode in self.modes:
            if mode == "values":
                payload: dict[str, Any] = {"messages": [message]}
            else:
                payload = {node: {"messages": [message]}}
            chunks.append(self.pack((), mode, payload))
        return chunks


def _updates(mode: str, payload: Any) -> list[Any]:
    if mode == "values":
        return [payload]
    return list(payload.values()) if isinstance(payload, dict) else []


def _payload_messages(mode: str, payload: Any) -> list[Any]:
    found: list[Any] = []
    for update in _updates(mode, payload):
        if isinstance(update, dict) and "messages" in update:
            value = update["messages"]
            found.extend(value if isinstance(value, (list, tuple)) else [value])
    return found


def _rewrite_payload(mode: str, payload: Any, rewrite: Callable[[Any], Any]) -> Any:
    def fix(update: Any) -> Any:
        if not isinstance(update, dict) or "messages" not in update:
            return update
        value = update["messages"]
        copied = dict(update)
        if isinstance(value, (list, tuple)):
            copied["messages"] = [rewrite(message) for message in value]
        else:
            copied["messages"] = rewrite(value)
        return copied

    if mode == "values":
        return fix(payload)
    if isinstance(payload, dict):
        return {node: fix(update) for node, update in payload.items()}
    return payload


class GuardedGraph:
    """Wraps a compiled LangGraph graph. Input is checked before the graph runs.

    ``stream`` and ``astream`` buffer the whole run, check the final answer,
    then yield. Only ``stream_mode`` ``"values"`` and ``"updates"`` are
    supported, because token streaming would reach the caller before the
    output check.
    """

    def __init__(self, graph: Any, policy: GuardPolicy, engine: GuardEngine) -> None:
        self.graph = graph
        self.policy = policy
        self.engine = engine

    def __getattr__(self, name: str) -> Any:
        return getattr(self.graph, name)

    def invoke(self, data: Any, config: Any = None, **kwargs: Any) -> Any:
        prepared, blocked = self._prepare_input(data)
        if blocked:
            return prepared
        return self._guard_output(self.graph.invoke(prepared, config, **kwargs))

    async def ainvoke(self, data: Any, config: Any = None, **kwargs: Any) -> Any:
        prepared, blocked = self._prepare_input(data)
        if blocked:
            return prepared
        return self._guard_output(await self.graph.ainvoke(prepared, config, **kwargs))

    def stream(self, data: Any, config: Any = None, **kwargs: Any) -> Iterator[Any]:
        layout = self._stream_layout(kwargs)
        prepared, blocked = self._prepare_input(data)
        if blocked:
            yield from layout.refusal(self.policy.refusal_message, "input_guard")
            return
        chunks = list(self.graph.stream(prepared, config, **kwargs))
        yield from self._guard_chunks(chunks, layout)

    async def astream(self, data: Any, config: Any = None, **kwargs: Any) -> AsyncIterator[Any]:
        layout = self._stream_layout(kwargs)
        prepared, blocked = self._prepare_input(data)
        if blocked:
            for chunk in layout.refusal(self.policy.refusal_message, "input_guard"):
                yield chunk
            return
        chunks = [chunk async for chunk in self.graph.astream(prepared, config, **kwargs)]
        for chunk in self._guard_chunks(chunks, layout):
            yield chunk

    def _stream_layout(self, kwargs: dict[str, Any]) -> _StreamLayout:
        mode = kwargs.get("stream_mode")
        if mode is None:
            mode = getattr(self.graph, "stream_mode", None) or "values"
        multi = isinstance(mode, (list, tuple))
        modes = tuple(mode) if multi else (mode,)
        unsupported = [name for name in modes if name not in STREAM_MODES]
        if unsupported or not modes:
            raise ValueError(
                f"jevrail can only guard stream_mode 'values' or 'updates', got {list(modes)}. "
                "Token and custom streams reach the caller before the output check runs."
            )
        return _StreamLayout(modes=modes, multi=multi, subgraphs=bool(kwargs.get("subgraphs")))

    def _prepare_input(self, data: Any) -> tuple[Any, bool]:
        if isinstance(data, str):
            decision = self.engine.judge("input", data)
            if decision.action == "block":
                self._raise_if_needed(decision)
                return self._refusal_state(), True
            if decision.action == "redact":
                return decision.text, False
            return data, False
        if not isinstance(data, dict) or "messages" not in data:
            return data, False
        messages = list(data["messages"])
        found = latest_text(messages, {"user", "tool"})
        if found is None:
            return data, False
        index, message, text = found
        decision = self.engine.judge("input", text)
        if decision.action == "block":
            self._raise_if_needed(decision)
            return self._refusal_state(), True
        if decision.action == "redact":
            messages[index] = with_content(message, decision.text)
            copied = dict(data)
            copied["messages"] = messages
            return copied, False
        return data, False

    def _guard_output(self, result: Any) -> Any:
        if not isinstance(result, dict) or "messages" not in result:
            return result
        messages = list(result["messages"])
        found = latest_text(messages, {"assistant"})
        if found is None:
            return result
        index, message, text = found
        decision = self.engine.judge("output", text)
        if decision.action in {"allow", "flag"}:
            return result
        if decision.action == "block":
            self._raise_if_needed(decision)
            messages[index] = {"role": "assistant", "content": decision.refusal}
        else:
            messages[index] = with_content(message, decision.text)
        copied = dict(result)
        copied["messages"] = messages
        return copied

    def _guard_chunks(self, chunks: list[Any], layout: _StreamLayout) -> list[Any]:
        answer = None
        for chunk in chunks:
            namespace, mode, payload = layout.unpack(chunk)
            if namespace:
                continue
            for message in _payload_messages(mode, payload):
                if message_role(message) == "assistant":
                    answer = message_text(message)
        if answer is None:
            return chunks
        decision = self.engine.judge("output", answer)
        if decision.action == "block":
            self._raise_if_needed(decision)
            return layout.refusal(decision.refusal or self.policy.refusal_message, "output_guard")
        if decision.action != "redact":
            return chunks

        def rewrite(message: Any) -> Any:
            if message_role(message) == "assistant" and message_text(message) == answer:
                return with_content(message, decision.text)
            return message

        guarded = []
        for chunk in chunks:
            namespace, mode, payload = layout.unpack(chunk)
            guarded.append(layout.pack(namespace, mode, _rewrite_payload(mode, payload, rewrite)))
        return guarded

    def _refusal_state(self) -> dict[str, Any]:
        return {"messages": [{"role": "assistant", "content": self.policy.refusal_message}]}

    def _raise_if_needed(self, decision: Any) -> None:
        if self.policy.on_block == "raise":
            raise GuardrailBlocked(decision)


def guard(
    graph: Any,
    policy: GuardPolicy | None = None,
    engine: GuardEngine | None = None,
    classifier: object | None = None,
) -> GuardedGraph:
    """Check the latest user message before ``graph`` runs and the final answer after."""
    active = policy or GuardPolicy.default()
    return GuardedGraph(graph, active, engine or GuardEngine(active, classifier))
