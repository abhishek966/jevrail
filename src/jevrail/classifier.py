"""Live Jev adapter. Tests inject their own classifier and never import this."""

from __future__ import annotations


class TypeSafeJevClassifier:
    """Score many yes/no questions in one TypeSafe Jev call."""

    def __init__(self) -> None:
        from langchain_typesafe import Noul, TypeSafeClassifier

        self._noul = Noul
        self._classifier = TypeSafeClassifier()

    def score(self, text: str, questions: dict[str, str]) -> dict[str, float]:
        response = self._classifier.invoke(
            state=text,
            questions={
                name: self._noul(instructions=instructions)
                for name, instructions in questions.items()
            },
        )
        nouls = getattr(response, "nouls", {}) or {}
        scores: dict[str, float] = {}
        for name in questions:
            item = nouls[name]
            value = getattr(item, "noul", item)
            scores[name] = float(value)
        return scores
