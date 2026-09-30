from __future__ import annotations

import logging
from dataclasses import dataclass, field

from jevrail.checks import CHECKS, SCANNER_LABELS
from jevrail.policy import GuardPolicy
from jevrail.scanners import Span, find_sensitive, redact

logger = logging.getLogger("jevrail")

_RANK = {"allow": 0, "flag": 1, "redact": 2, "block": 3}


class GuardrailBlocked(Exception):
    """Raised when ``on_block='raise'`` and a check blocks the turn."""

    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        super().__init__(decision.refusal or "Blocked by jevrail")


@dataclass(frozen=True)
class Hit:
    check_id: str
    score: float
    threshold: float
    action: str
    detail: str = ""


@dataclass(frozen=True)
class Decision:
    action: str
    stage: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refusal: str | None = None

    @property
    def blocked(self) -> bool:
        return self.action == "block"


class GuardEngine:
    """Ask Jev one batched question set per stage, then apply the policy."""

    def __init__(self, policy: GuardPolicy | None = None, classifier: object | None = None) -> None:
        self.policy = policy or GuardPolicy.default()
        self.classifier = classifier

    def judge(self, stage: str, text: str, tool_name: str | None = None) -> Decision:
        checks = list(self.policy.stages.get(stage) or [])
        if (
            stage == "tool_call"
            and tool_name
            and tool_name in self.policy.denied_tools
            and stage in self.policy.stages
        ):
            return self._blocked(
                stage,
                [
                    Hit(
                        check_id="denied_tool",
                        score=1.0,
                        threshold=1.0,
                        action="block",
                        detail=tool_name,
                    )
                ],
            )
        if not checks:
            return Decision(action="allow", stage=stage, text=text)

        hits: list[Hit] = []
        redact_spans: list[Span] = []
        questions: dict[str, str] = {}
        for check_id in checks:
            check = CHECKS[check_id]
            if check_id == "topic_scope":
                if not self.policy.allowed_topics:
                    continue
                topics = ", ".join(self.policy.allowed_topics)
                questions[check_id] = check.instructions.format(topics=topics)
                continue
            if check.scanner:
                matched = find_sensitive(text, SCANNER_LABELS[check.scanner])
                if matched:
                    redact_spans.extend(matched)
                    action = self.policy.action_for(check_id)
                    hits.append(
                        Hit(
                            check_id=check_id,
                            score=1.0,
                            threshold=self.policy.threshold_for(check_id),
                            action=action,
                            detail=",".join(sorted({span.label for span in matched})),
                        )
                    )
            questions[check_id] = check.instructions

        if questions:
            try:
                scores = self._score(text, questions)
            except Exception:
                logger.warning("jevrail classifier failed on stage %s", stage, exc_info=True)
                if self.policy.fail_mode == "closed":
                    return self._blocked(
                        stage,
                        hits
                        + [
                            Hit(
                                check_id="classifier_error",
                                score=1.0,
                                threshold=0.0,
                                action="block",
                                detail="classifier unavailable",
                            )
                        ],
                    )
                scores = {}
        else:
            scores = {}

        already = {hit.check_id for hit in hits}
        for check_id, score in scores.items():
            threshold = self.policy.threshold_for(check_id)
            if score < threshold or check_id in already:
                continue
            action = self.policy.action_for(check_id)
            check = CHECKS[check_id]
            if action == "redact" and check.scanner and not any(
                span.label in SCANNER_LABELS[check.scanner] for span in redact_spans
            ):
                action = "block"
            hits.append(
                Hit(
                    check_id=check_id,
                    score=float(score),
                    threshold=threshold,
                    action=action,
                )
            )

        if not hits:
            return Decision(action="allow", stage=stage, text=text)

        action = max(hits, key=lambda hit: _RANK[hit.action]).action
        logger.info(
            "jevrail stage=%s action=%s hits=%s",
            stage,
            action,
            [(hit.check_id, hit.score) for hit in hits],
        )
        if action == "block":
            return self._blocked(stage, hits)
        if action == "redact":
            return Decision(action="redact", stage=stage, text=redact(text, redact_spans), hits=hits)
        if action == "flag":
            return Decision(action="flag", stage=stage, text=text, hits=hits)
        return Decision(action="allow", stage=stage, text=text, hits=hits)

    def judge_or_raise(self, stage: str, text: str, tool_name: str | None = None) -> Decision:
        decision = self.judge(stage, text, tool_name=tool_name)
        if decision.action == "block" and self.policy.on_block == "raise":
            raise GuardrailBlocked(decision)
        return decision

    def _blocked(self, stage: str, hits: list[Hit]) -> Decision:
        return Decision(
            action="block",
            stage=stage,
            text=self.policy.refusal_message,
            hits=hits,
            refusal=self.policy.refusal_message,
        )

    def _score(self, text: str, questions: dict[str, str]) -> dict[str, float]:
        classifier = self.classifier
        if classifier is None:
            from jevrail.classifier import TypeSafeJevClassifier

            classifier = TypeSafeJevClassifier()
            self.classifier = classifier
        return classifier.score(text, questions)
