from pathlib import Path

import pytest

from jevrail import GuardPolicy
from jevrail.policy import DEFAULT_STAGES


def test_default_policy_covers_input_and_output_only():
    policy = GuardPolicy.default()
    assert policy.stages["input"] == DEFAULT_STAGES["input"]
    assert "prompt_injection" in policy.stages["input"]
    assert "content_safety" in policy.stages["output"]
    assert "tool_call" not in policy.stages
    assert policy.fail_mode == "closed"
    assert policy.on_block == "refuse"


def test_yaml_round_trip(tmp_path: Path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        "stages:\n"
        "  input: [prompt_injection]\n"
        "  output: [pii]\n"
        "thresholds:\n"
        "  prompt_injection: 0.9\n"
        "actions:\n"
        "  pii: redact\n"
        "on_block: raise\n"
        "fail_mode: open\n"
        "allowed_topics: [billing]\n"
        "denied_tools: [bash]\n",
        encoding="utf-8",
    )
    policy = GuardPolicy.from_yaml(path)
    assert policy.stages["input"] == ["prompt_injection"]
    assert policy.thresholds["prompt_injection"] == 0.9
    assert policy.actions["pii"] == "redact"
    assert policy.on_block == "raise"
    assert policy.fail_mode == "open"
    assert policy.allowed_topics == ("billing",)
    assert policy.denied_tools == ("bash",)


def test_unknown_check_is_rejected():
    with pytest.raises(ValueError):
        GuardPolicy(stages={"input": ["not_a_real_check"]})
