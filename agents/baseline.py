"""
Baseline agent for Condition A: a single LLM call that does everything
in one shot. No decomposition, no workers, no validation.

This is the "dumb" baseline that any sensible system must beat. If
Conditions B and C don't improve on this, the orchestrator architecture
isn't pulling its weight — and the paper needs a different story.
"""

from llm_client import call_llm_json


BASELINE_SYSTEM = """You are a legal contract analyst."""


BASELINE_PROMPT_TEMPLATE = """Analyse the following contract and identify
all clauses that match any of the provided categories. For each match,
return the exact verbatim text from the contract, the category it
belongs to, and a risk assessment.

=== CATEGORIES ===

{categories_block}

=== CONTRACT ===

{contract_text}

=== OUTPUT FORMAT ===

Return ONLY a JSON array, no preamble:

    [
      {{
        "category": "<category name>",
        "text": "<exact verbatim text from the contract>",
        "confidence": "high" | "medium" | "low",
        "risk_level": "low" | "medium" | "high"
      }},
      ...
    ]

If no matches are found, return [].

=== JSON OUTPUT ===
"""


class BaselineAgent:

    def __init__(self, categories: list[dict]):
        self.categories = categories
        self._categories_block = "\n".join(
            f'- "{c["name"]}": {c["description"]}' for c in categories
        )

    def review(self, contract_text: str, contract_id: str = "") -> dict:
        """
        Single-shot review. Return the same shape as the orchestrator,
        so the evaluator can handle both.
        """
        print(f"[baseline] reviewing {contract_id}")
        prompt = BASELINE_PROMPT_TEMPLATE.format(
            categories_block=self._categories_block,
            contract_text=contract_text,
        )
        try:
            result = call_llm_json(
                prompt=prompt,
                system=BASELINE_SYSTEM,
                tag="baseline",
            )
        except Exception as e:
            print(f"  [baseline] call failed: {e}")
            result = []

        if not isinstance(result, list):
            result = []

        # Normalise clause shape to match orchestrator output.
        clauses = []
        for c in result:
            clauses.append({
                "category": c.get("category", ""),
                "classified_category": c.get("category", ""),
                "text": c.get("text", ""),
                "confidence": c.get("confidence", "medium"),
                "risk_level": c.get("risk_level", "low"),
                "risk_factors": [],
                "risk_reasoning": "",
            })

        return {
            "contract_id": contract_id,
            "clauses": clauses,
        }
