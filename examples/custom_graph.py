"""Place guard nodes on the edges of a custom graph.

``guard(compiled)`` checks the user message and the final answer.
``input_guard`` and ``output_guard`` are for a specific edge.
"""

import os

from jevrail import GuardPolicy, guard, input_guard, output_guard


def build():
    from typing import Annotated

    from typing_extensions import TypedDict
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    policy = GuardPolicy.default()

    def reply(state: State) -> dict:
        return {"messages": [{"role": "assistant", "content": "Done."}]}

    graph = StateGraph(State)
    graph.add_node("input_guard", input_guard(policy))
    graph.add_node("reply", reply)
    graph.add_node("output_guard", output_guard(policy))
    graph.add_edge(START, "input_guard")
    graph.add_edge("input_guard", "reply")
    graph.add_edge("reply", "output_guard")
    graph.add_edge("output_guard", END)
    return guard(graph.compile(), policy)


if __name__ == "__main__":
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("Set TYPESAFE_API_KEY before invoking the graph.")
    app = build()
    print(app.invoke({"messages": [{"role": "user", "content": "Hello"}]}))
