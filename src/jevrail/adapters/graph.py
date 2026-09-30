from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from jevrail.engine import GuardEngine, GuardrailBlocked
from jevrail.messages import latest_text, with_content
from jevrail.policy import GuardPolicy


class GuardedGraph:
    """Wraps a compiled LangGraph graph. Input is checked before the graph runs."""

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
        yield from self._stream(data, config, **kwargs)

    async def astream(self, data: Any, config: Any = None, **kwargs: Any) -> AsyncIterator[Any]:
        prepared, blocked = self._prepare_input(data)
        if blocked:
            yield prepared
            return
        chunks = []
        async for chunk in self.graph.astream(prepared, config, **kwargs):
            chunks.append(chunk)
        guarded = self._guard_chunks(chunks)
        for chunk in guarded:
            yield chunk

    def _stream(self, data: Any, config: Any = None, **kwargs: Any) -> Iterator[Any]:
        prepared, blocked = self._prepare_input(data)
        if blocked:
            yield prepared
            return
        chunks = list(self.graph.stream(prepared, config, **kwargs))
        yield from self._guard_chunks(chunks)

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

    def _guard_chunks(self, chunks: list[Any]) -> list[Any]:
        assistant = None
        for chunk in chunks:
            if isinstance(chunk, dict) and "messages" in chunk:
                found = latest_text(list(chunk["messages"]), {"assistant"})
                if found is not None:
                    assistant = found[2]
        if assistant is None:
            return chunks
        decision = self.engine.judge("output", assistant)
        if decision.action == "block":
            self._raise_if_needed(decision)
            return [self._refusal_state()]
        return chunks

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
