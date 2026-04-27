"""
Classifier worker agent.

Job: given a clause text and the full category taxonomy, re-verify or
refine the category label assigned by the extractor.

Why have both an extractor AND a classifier?
    - The extractor optimises for recall (find candidate clauses).
    - The classifier optimises for precision (confirm the label).
    - Separating the two jobs is a standard pattern in IR/NLP pipelines
      (retrieve + rerank) and fits the orchestrator-workers design
      nicely.

The classifier sees only the clause text (not the full chunk) so it
makes its decision based solely on the clause content. This catches
cases where the extractor labelled something wrong because of
surrounding context.
"""

from llm_client import call_llm_json


SYSTEM_PROMPT = """You are a legal text classification expert. You
classify contract clauses into predefined categories. You follow strict
definitions and do not invent new categories."""


CLASSIFIER_PROMPT_TEMPLATE = """Classify the following contract clause
into exactly ONE of the provided categories. If it does not fit any
category, return "OTHER".

=== CATEGORIES ===

{categories_block}

=== CLAUSE TO CLASSIFY ===

"{clause_text}"

=== INSTRUCTIONS ===

1. Read the clause carefully.
2. Pick the single best-matching category name. If none applies, return
   "OTHER".
3. Explain your reasoning in one sentence.
4. Assign a confidence level: "high" / "medium" / "low".

=== OUTPUT FORMAT ===

Return ONLY a JSON object:

    {{
      "category": "<category name or OTHER>",
      "reasoning": "<one-sentence justification>",
      "confidence": "high" | "medium" | "low"
    }}

=== JSON OUTPUT ===
"""


class Classifier:

    def __init__(self, categories: list[dict]):
        self.categories = categories
        self.valid_names = {c["name"] for c in categories} | {"OTHER"}
        self._categories_block = self._format_categories(categories)

    def _format_categories(self, categories: list[dict]) -> str:
        lines = []
        for c in categories:
            lines.append(f'- "{c["name"]}": {c["description"]}')
        return "\n".join(lines)

    def classify(self, clause_text: str) -> dict:
        """
        Return {"category", "reasoning", "confidence"}.

        If the model returns an unknown category, we coerce to "OTHER"
        and lower the confidence — this is a mini rule-based safeguard.
        """
        prompt = CLASSIFIER_PROMPT_TEMPLATE.format(
            categories_block=self._categories_block,
            clause_text=clause_text,
        )
        result = call_llm_json(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            tag="classifier",
        )

        # Defensive: validate category name.
        if not isinstance(result, dict) or "category" not in result:
            return {"category": "OTHER", "reasoning": "malformed output",
                    "confidence": "low"}

        if result["category"] not in self.valid_names:
            print(f"  [classifier] unknown category {result['category']!r}, "
                  f"coercing to OTHER")
            result["category"] = "OTHER"
            result["confidence"] = "low"

        return result
