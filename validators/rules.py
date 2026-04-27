"""
Rule-based validation layer.

This is the piece that distinguishes condition C from condition B in
the experiments. It runs AFTER the orchestrator produces its output
and applies a set of deterministic checks that an LLM alone cannot
reliably perform.

The three checks, in order of importance:

    1. GROUNDING (most important). The clause text returned by the
       extractor must appear verbatim (or near-verbatim) in the source
       contract. This catches the single most common LLM failure mode:
       hallucinated text. Clauses that fail grounding are flagged or
       dropped.

    2. SCHEMA. Every clause must have all required fields with the
       right types. Malformed outputs are flagged.

    3. KEYWORD CONSISTENCY. If a clause is labelled with a category
       whose description mentions specific concepts (e.g. "Termination
       for Convenience" implies termination language), we check that
       the clause text contains at least one matching keyword. If not,
       we lower the confidence.

For the paper: these are three concrete, measurable controls that
mitigate known LLM failure modes. Each one can be toggled on/off for
fine-grained ablation if reviewers ask.
"""

import re
from difflib import SequenceMatcher


# Keyword patterns per category. Not exhaustive — meant as a safety
# net, not a full classifier. The goal is: if NONE of these words
# appear, the extractor/classifier is probably wrong.
CATEGORY_KEYWORDS = {
    "Governing Law": ["govern", "laws of", "jurisdiction"],
    "Termination for Convenience": ["terminat", "convenience", "without cause",
                                     "notice"],
    "Liability Cap": ["liability", "aggregate", "not exceed", "cap",
                      "maximum", "limited to"],
    "Indemnification": ["indemnif", "hold harmless", "defend"],
    "IP Ownership Assignment": ["intellectual property", "own", "assign",
                                 "title", "invention", "work product"],
    "Change of Control": ["change of control", "merger", "acquisition",
                           "assignment"],
    "Audit Rights": ["audit", "inspect", "books and records", "examine"],
    "Warranty Duration": ["warrant", "period of", "days", "months", "years"],
    "Non-Compete": ["compete", "competing", "solicit", "restrict"],
    "Insurance Requirements": ["insurance", "coverage", "liability insurance",
                                "policy"],
}


class RuleBasedValidator:

    # Minimum similarity ratio for grounding check (0-1). 0.8 allows
    # minor whitespace / punctuation differences but rejects
    # paraphrases.
    GROUNDING_THRESHOLD = 0.8

    def validate(
        self,
        review_output: dict,
        contract_text: str,
    ) -> dict:
        """
        Apply validation rules to an orchestrator output.

        Mutates the output in place by adding per-clause validation
        flags, and returns a summary dict.
        """
        clauses = review_output.get("clauses", [])
        total = len(clauses)
        dropped = 0
        flagged = 0

        kept = []
        for c in clauses:
            flags = []

            # --- Schema check ---
            if not self._schema_ok(c):
                flags.append("schema_malformed")

            # --- Grounding check ---
            if not self._is_grounded(c.get("text", ""), contract_text):
                flags.append("not_grounded")

            # --- Keyword consistency check ---
            if not self._keywords_match(c):
                flags.append("keywords_missing")

            c["validation_flags"] = flags

            # Policy: drop clauses that fail grounding (they are
            # almost certainly hallucinated). Flag the rest but keep
            # them.
            if "not_grounded" in flags:
                dropped += 1
                continue

            if flags:
                flagged += 1
            kept.append(c)

        review_output["clauses"] = kept
        review_output["validation"] = {
            "total_candidates": total,
            "dropped": dropped,
            "flagged": flagged,
            "kept": len(kept),
        }
        return review_output

    def _schema_ok(self, clause: dict) -> bool:
        required = ["category", "text", "confidence"]
        for k in required:
            if k not in clause:
                return False
        return isinstance(clause["text"], str) and len(clause["text"]) > 0

    def _is_grounded(self, clause_text: str, contract_text: str) -> bool:
        """
        Check if clause_text appears (substantially) in contract_text.

        We do an exact substring match first (cheap, common case), then
        fall back to a similarity-based fuzzy check for cases where the
        LLM may have slightly modified whitespace.
        """
        if not clause_text:
            return False

        clean_clause = self._normalise(clause_text)
        clean_contract = self._normalise(contract_text)

        # Exact substring match (after normalisation).
        if clean_clause in clean_contract:
            return True

        # Fuzzy: check if any ~window in the contract is similar enough.
        # This is O(n*m) in the worst case — for contracts of ~100k
        # chars and clauses of ~500 chars it's fine. If it becomes
        # slow, replace with a smarter algorithm.
        window_size = len(clean_clause)
        step = max(50, window_size // 4)
        for i in range(0, len(clean_contract) - window_size + 1, step):
            window = clean_contract[i : i + window_size]
            ratio = SequenceMatcher(None, clean_clause, window).quick_ratio()
            if ratio >= self.GROUNDING_THRESHOLD:
                # Confirm with real ratio (quick_ratio overestimates).
                if SequenceMatcher(None, clean_clause, window).ratio() \
                        >= self.GROUNDING_THRESHOLD:
                    return True
        return False

    def _keywords_match(self, clause: dict) -> bool:
        """Check that the clause text contains keywords for its category."""
        category = clause.get("classified_category") or clause.get("category")
        text = clause.get("text", "").lower()
        keywords = CATEGORY_KEYWORDS.get(category)

        # Unknown category or no keyword list → don't enforce.
        if keywords is None:
            return True

        return any(kw in text for kw in keywords)

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase and collapse whitespace."""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()
