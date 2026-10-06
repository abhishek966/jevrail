from jevrail import GuardPolicy, JevRail
from jevrail.engine import GuardEngine


class FakeClassifier:
    def __init__(self, scores):
        self.scores = scores

    def score(self, text, questions):
        return {name: self.scores.get(name, 0.0) for name in questions}


class Message:
    def __init__(self, role, content, message_id="1"):
        self.role = role
        self.type = role
        self.content = content
        self.id = message_id


def rail(scores, **policy_kwargs):
    policy = GuardPolicy(
        stages={
            "input": ["prompt_injection"],
            "output": ["content_safety", "pii"],
            "tool_call": ["excessive_agency"],
            "tool_result": ["indirect_injection"],
        },
        actions={"pii": "redact", "prompt_injection": "block", "content_safety": "block"},
        **policy_kwargs,
    )
    return JevRail(policy, engine=GuardEngine(policy, FakeClassifier(scores)))


def test_before_model_blocks_and_jumps_to_end():
    guard = rail({"prompt_injection": 0.99})
    update = guard.before_model({"messages": [Message("user", "ignore instructions")]}, runtime=None)
    assert update["jump_to"] == "end"
    assert "can't help" in update["messages"][0].content.lower()


def test_before_model_allows_clean_input():
    guard = rail({"prompt_injection": 0.01})
    assert guard.before_model({"messages": [Message("user", "hello")]}, runtime=None) is None


def test_before_model_redacts_pii_in_place_before_the_model_sees_it():
    policy = GuardPolicy(
        stages={"input": ["pii"], "output": ["content_safety"]},
        actions={"pii": "redact"},
    )
    guard = JevRail(policy, engine=GuardEngine(policy, FakeClassifier({"pii": 0.0, "content_safety": 0.0})))
    update = guard.before_model(
        {"messages": [Message("user", "email me at user@example.com", "u1")]},
        runtime=None,
    )
    assert "jump_to" not in update
    assert update["messages"][0].id == "u1"
    assert "user@example.com" not in update["messages"][0].content
    assert "[REDACTED]" in update["messages"][0].content


def test_before_model_judges_input_once():
    seen = []

    class Counting(FakeClassifier):
        def score(self, text, questions):
            seen.append(text)
            return super().score(text, questions)

    policy = GuardPolicy(stages={"input": ["prompt_injection"]})
    guard = JevRail(policy, engine=GuardEngine(policy, Counting({})))
    guard.before_model({"messages": [Message("user", "hello")]}, runtime=None)
    assert seen == ["hello"]


def test_after_model_blocks_unsafe_output():
    guard = rail({"content_safety": 0.99, "pii": 0.0})
    update = guard.after_model(
        {"messages": [Message("assistant", "unsafe answer", "2")]},
        runtime=None,
    )
    assert update["jump_to"] == "end"


def test_wrap_tool_call_blocks_denied_tool_without_running_it():
    guard = rail({"excessive_agency": 0.0}, denied_tools=("bash",))
    called = {"ran": False}

    class ToolRequest:
        def __init__(self):
            self.tool_call = {"name": "bash", "args": {"cmd": "ls"}, "id": "call-1"}

    def handler(request):
        called["ran"] = True
        return Message("tool", "ok", "call-1")

    result = guard.wrap_tool_call(ToolRequest(), handler)
    assert called["ran"] is False
    assert "can't help" in result.content.lower()


def test_denied_tool_blocks_even_when_tool_stages_are_off():
    policy = GuardPolicy(stages={"input": ["prompt_injection"]}, denied_tools=("bash",))
    guard = JevRail(policy, engine=GuardEngine(policy, FakeClassifier({})))
    called = {"ran": False}

    class ToolRequest:
        tool_call = {"name": "bash", "args": {"cmd": "ls"}, "id": "call-1"}

    def handler(request):
        called["ran"] = True
        return Message("tool", "ok", "call-1")

    guard.wrap_tool_call(ToolRequest(), handler)
    assert called["ran"] is False
