"""Build a guarded agent with ``create_agent``.

Requires the ``langgraph`` extra and ``TYPESAFE_API_KEY``.
"""

import os

from jevrail import GuardPolicy, JevRail


def build_agent():
    from langchain.agents import create_agent

    policy = GuardPolicy(
        stages={
            "input": ["prompt_injection", "jailbreak", "content_safety"],
            "output": ["content_safety", "pii", "secrets", "system_prompt_leak"],
        },
        actions={"pii": "redact", "secrets": "redact"},
    )
    return create_agent(
        os.environ.get("JEVRAIL_MODEL", "openai:gpt-4o-mini"),
        tools=[],
        middleware=[JevRail(policy)],
    )


if __name__ == "__main__":
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("Set TYPESAFE_API_KEY before invoking the agent.")
    agent = build_agent()
    print(agent.invoke({"messages": [{"role": "user", "content": "Summarize the refund policy."}]}))
