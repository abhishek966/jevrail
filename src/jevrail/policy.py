from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jevrail.checks import ACTIONS, CHECKS, STAGES

DEFAULT_STAGES: dict[str, list[str]] = {
    "input": [
        "prompt_injection",
        "jailbreak",
        "content_safety",
        "system_prompt_extraction",
    ],
    "output": [
        "content_safety",
        "pii",
        "secrets",
        "system_prompt_leak",
        "data_exfiltration",
    ],
}

REFUSAL = "I can't help with that request."


@dataclass
class GuardPolicy:
    """Which checks run on which stage, and what to do when one hits.

    ``stages`` replaces the defaults when you pass it. A stage you omit is off.
    An empty list turns that stage off explicitly.
    """

    stages: dict[str, list[str]] | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    actions: dict[str, str] = field(default_factory=dict)
    on_block: str = "refuse"
    fail_mode: str = "closed"
    allowed_topics: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    refusal_message: str = REFUSAL

    def __post_init__(self) -> None:
        source = DEFAULT_STAGES if self.stages is None else self.stages
        self.stages = {stage: list(checks) for stage, checks in source.items()}
        self.allowed_topics = tuple(self.allowed_topics)
        self.denied_tools = tuple(self.denied_tools)
        self.thresholds = {key: float(value) for key, value in self.thresholds.items()}
        if self.on_block not in {"refuse", "raise"}:
            raise ValueError("on_block must be 'refuse' or 'raise'")
        if self.fail_mode not in {"closed", "open"}:
            raise ValueError("fail_mode must be 'closed' or 'open'")
        unknown_stages = set(self.stages) - STAGES
        if unknown_stages:
            raise ValueError(f"Unknown stages: {sorted(unknown_stages)}")
        for stage, checks in self.stages.items():
            for check_id in checks:
                if check_id not in CHECKS:
                    raise ValueError(f"Unknown check {check_id!r} on stage {stage!r}")
        for check_id in list(self.thresholds) + list(self.actions):
            if check_id not in CHECKS:
                raise ValueError(f"Unknown check {check_id!r}")
        for action in self.actions.values():
            if action not in ACTIONS:
                raise ValueError(f"Unknown action {action!r}")

    @classmethod
    def default(cls) -> GuardPolicy:
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> GuardPolicy:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            stages=raw.get("stages"),
            thresholds=raw.get("thresholds") or {},
            actions=raw.get("actions") or {},
            on_block=raw.get("on_block", "refuse"),
            fail_mode=raw.get("fail_mode", "closed"),
            allowed_topics=tuple(raw.get("allowed_topics") or ()),
            denied_tools=tuple(raw.get("denied_tools") or ()),
            refusal_message=raw.get("refusal_message", REFUSAL),
        )

    def threshold_for(self, check_id: str) -> float:
        if check_id in self.thresholds:
            return self.thresholds[check_id]
        return CHECKS[check_id].default_threshold

    def action_for(self, check_id: str) -> str:
        if check_id in self.actions:
            return self.actions[check_id]
        return CHECKS[check_id].default_action
