"""
Risk scorer worker agent.

Job: given a clause text and its category, estimate the contractual
risk it represents for the reviewing party.

This is the agent that turns the system from "clause classifier" into
"decision support system". A contract manager doesn't just want to know
"this is a liability cap" — they want to know "this liability cap is
unusually low and exposes you to X".

For the paper:
    - The risk scorer is the bridge between technical NLP output and
      managerial decision support. This makes the system relevant to
      the industrial audience at the Summer School, not just to legal
      NLP researchers.
    - Citing reasoning (the "explanation" field) is critical for
      human-in-the-loop interpretability.
"""

from llm_client import call_llm_json


SYSTEM_PROMPT = """You are a senior contract risk analyst advising the
buyer / client side of commercial contracts. You assess clauses for
commercial and legal risk. You are cautious but not alarmist."""


RISK_PROMPT_TEMPLATE = """Assess the risk of the following contract
clause from the perspective of the buyer / client party.

=== CATEGORY ===

{category}

=== CLAUSE ===

"{clause_text}"

=== RISK ASSESSMENT CRITERIA ===

- "low": standard / market-typical wording, balanced allocation of
  obligations, reasonable thresholds
- "medium": some asymmetry or unusual wording, worth flagging to the
  contract manager for review
- "high": strongly unfavourable, unusual, ambiguous in a way that could
  expose the reviewing party to significant loss, missing caps or
  carve-outs where standard practice expects them

=== INSTRUCTIONS ===

1. Read the clause in context of its category.
2. Assign a risk level (low / medium / high).
3. Identify specific risk factors (e.g. "uncapped liability", "automatic
   renewal without notice", "broad indemnification").
4. Write a short reasoning sentence.

=== OUTPUT FORMAT ===

Return ONLY a JSON object:

    {{
      "risk_level": "low" | "medium" | "high",
      "risk_factors": ["<factor 1>", "<factor 2>", ...],
      "reasoning": "<one-sentence justification>"
    }}

=== JSON OUTPUT ===
"""


class RiskScorer:

    def score(self, clause_text: str, category: str) -> dict:
        """
        Return {"risk_level", "risk_factors", "reasoning"}.
        """
        prompt = RISK_PROMPT_TEMPLATE.format(
            category=category,
            clause_text=clause_text,
        )
        result = call_llm_json(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            tag="risk_scorer",
        )

        # Defensive defaults.
        if not isinstance(result, dict):
            return {"risk_level": "low", "risk_factors": [],
                    "reasoning": "malformed output"}

        result.setdefault("risk_level", "low")
        result.setdefault("risk_factors", [])
        result.setdefault("reasoning", "")

        if result["risk_level"] not in ("low", "medium", "high"):
            result["risk_level"] = "low"

        return result
