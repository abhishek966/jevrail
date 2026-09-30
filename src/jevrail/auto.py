"""Opt-in patch. Importing this module guards ``create_agent`` and messages graphs.

Import it before you build the agent:

    import jevrail.auto
"""

from __future__ import annotations

from typing import Any

from jevrail.adapters.graph import guard
from jevrail.adapters.middleware import JevRail
from jevrail.policy import GuardPolicy

_INSTALLED = False
_SAVED: list[tuple[Any, str, Any]] = []


def _load_agents() -> Any:
    import langchain.agents as agents

    return agents


def _load_state_graph() -> Any:
    from langgraph.graph.state import StateGraph

    return StateGraph


def _policy_from_env() -> GuardPolicy:
    import os

    path = os.environ.get("JEVRAIL_POLICY")
    if not path:
        return GuardPolicy.default()
    policy = GuardPolicy.from_yaml(path)
    fail_mode = os.environ.get("JEVRAIL_FAIL_MODE")
    on_block = os.environ.get("JEVRAIL_ON_BLOCK")
    if fail_mode or on_block:
        return GuardPolicy(
            stages=policy.stages,
            thresholds=policy.thresholds,
            actions=policy.actions,
            on_block=on_block or policy.on_block,
            fail_mode=fail_mode or policy.fail_mode,
            allowed_topics=policy.allowed_topics,
            denied_tools=policy.denied_tools,
            refusal_message=policy.refusal_message,
        )
    return policy


def install(policy: GuardPolicy | None = None) -> None:
    """Patch LangChain ``create_agent`` and LangGraph ``StateGraph.compile`` once."""
    global _INSTALLED
    if _INSTALLED:
        return
    active = policy or _policy_from_env()
    _patch_create_agent(active)
    _patch_compile(active)
    _INSTALLED = True


def uninstall() -> None:
    """Restore patched call sites. Tests use this so an import does not leak."""
    global _INSTALLED
    while _SAVED:
        obj, attr, original = _SAVED.pop()
        setattr(obj, attr, original)
    _INSTALLED = False


def _patch_create_agent(policy: GuardPolicy) -> None:
    try:
        agents = _load_agents()
    except ImportError:
        return
    original = agents.create_agent
    if getattr(original, "_jevrail_patched", False):
        return

    def create_agent(*args: Any, **kwargs: Any) -> Any:
        middleware = list(kwargs.get("middleware") or [])
        if not any(type(item).__name__ == "JevRail" for item in middleware):
            middleware.append(JevRail(policy))
        kwargs["middleware"] = middleware
        return original(*args, **kwargs)

    create_agent._jevrail_patched = True  # type: ignore[attr-defined]
    _SAVED.append((agents, "create_agent", original))
    agents.create_agent = create_agent


def _patch_compile(policy: GuardPolicy) -> None:
    try:
        state_graph = _load_state_graph()
    except ImportError:
        return
    original = state_graph.compile
    if getattr(original, "_jevrail_patched", False):
        return

    def compile(self: Any, *args: Any, **kwargs: Any) -> Any:
        graph = original(self, *args, **kwargs)
        if type(graph).__name__ == "GuardedGraph":
            return graph
        if _has_messages(self):
            return guard(graph, policy)
        return graph

    compile._jevrail_patched = True  # type: ignore[attr-defined]
    _SAVED.append((state_graph, "compile", original))
    state_graph.compile = compile


def _has_messages(builder: Any) -> bool:
    channels = getattr(builder, "channels", None) or {}
    if "messages" in channels:
        return True
    schema = getattr(builder, "state_schema", None)
    annotations = getattr(schema, "__annotations__", {}) if schema else {}
    return "messages" in annotations


install()
