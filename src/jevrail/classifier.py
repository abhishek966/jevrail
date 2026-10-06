"""Live Jev adapter. Tests inject their own classifier and never import this."""

from __future__ import annotations


class TypeSafeJevClassifier:
    """Score many yes/no questions in one TypeSafe Jev call.

    ``classifier`` is a configured ``langchain_typesafe.TypeSafeClassifier``.
    By default one is built from ``TYPESAFE_API_KEY`` and ``TYPESAFE_BASE_URL``.
    """

    def __init__(self, classifier: object | None = None) -> None:
        from langchain_typesafe import Noul, TypeSafeClassifier

        self._noul = Noul
        self._classifier = classifier if classifier is not None else TypeSafeClassifier()

    def score(self, text: str, questions: dict[str, str]) -> dict[str, float]:
        response = self._classifier.invoke(
            {
                "state": text,
                "questions": {
                    name: self._noul(instructions=instructions)
                    for name, instructions in questions.items()
                },
            }
        )
        nouls = response.nouls
        return {name: float(nouls[name].noul) for name in questions}
