"""
Orchestrator agent — the brain of the system.

Job: coordinate the four worker agents (segmenter, extractor,
classifier, risk scorer) to produce a complete contract review.

This is what makes the system a multi-agent orchestrator rather than a
single giant prompt. Design decisions embedded here:

    1. PLANNING. The orchestrator first decides which categories are
       relevant to this specific contract. This is dynamic task
       decomposition: a short contract about software licensing doesn't
       need a "Warranty Duration" pass, for example. (For the paper:
       this is the "Plan" step of the Plan-Execute-Evaluate pattern.)

    2. SEGMENT + FAN-OUT. The contract is segmented, then each chunk is
       sent to the extractor. (For simplicity we loop sequentially; you
       can parallelise with concurrent.futures if needed — note in the
       paper that parallelisation is an engineering optimisation, not a
       design change.)

    3. AGGREGATE. All extracted clauses are merged into a single list.
       Duplicates (same text, same category) are collapsed.

    4. REFINE. Each clause is sent to the classifier (verify/refine
       label) and the risk scorer (add risk metadata).

    5. FINAL OUTPUT. A structured object with all clauses, labels,
       risks, and provenance (which chunk they came from).

The orchestrator returns this object. The caller (experiment runner) is
responsible for passing it through the validator if the condition
requires it — this keeps the orchestrator cleanly separable.
"""

from llm_client import call_llm_json
from agents.segmenter import Segmenter
from agents.extractor import Extractor
from agents.classifier import Classifier
from agents.risk_scorer import RiskScorer


PLANNER_SYSTEM = """You are a contract triage expert. You quickly
identify which types of provisions a contract likely contains."""


PLANNER_PROMPT_TEMPLATE = """Given a short excerpt from the beginning
of a contract, predict which of the following clause categories are
likely to appear somewhere in the full document.

=== CATEGORIES ===

{categories_block}

=== CONTRACT EXCERPT (first ~2000 chars) ===

{excerpt}

=== INSTRUCTIONS ===

Return the category names that are LIKELY present. When in doubt,
include the category (recall is more important than precision here).

=== OUTPUT FORMAT ===

    {{
      "likely_categories": ["<name 1>", "<name 2>", ...]
    }}

=== JSON OUTPUT ===
"""


class Orchestrator:

    def __init__(self, categories: list[dict], use_planning: bool = True):
        """
        Args:
            categories: the list of all target categories.
            use_planning: if True, run a planning step to narrow down
                          categories per contract. If False, always
                          search for all categories (baseline mode).
        """
        self.all_categories = categories
        self.use_planning = use_planning

        self.segmenter = Segmenter()
        self.extractor = Extractor(categories)   # default; may be rebuilt
        self.classifier = Classifier(categories)
        self.risk_scorer = RiskScorer()

    def review(self, contract_text: str, contract_id: str = "") -> dict:
        """
        Full contract review pipeline.

        Returns a dict with the structured review output.
        """
        print(f"[orchestrator] starting review of {contract_id}")

        # --- 1. Planning ---
        if self.use_planning:
            relevant_categories = self._plan(contract_text)
        else:
            relevant_categories = list(self.all_categories)

        print(f"  planning: {len(relevant_categories)} categories selected")

        # Rebuild the extractor with only the relevant categories so the
        # prompt is focused.
        extractor = Extractor(relevant_categories)

        # --- 2. Segment ---
        chunks = self.segmenter.split(contract_text)
        print(f"  segmenter: produced {len(chunks)} chunks")

        # --- 3. Extract (fan-out over chunks) ---
        all_clauses = []
        for chunk in chunks:
            extracted = extractor.extract(chunk["text"])
            for c in extracted:
                c["chunk_id"] = chunk["chunk_id"]
                all_clauses.append(c)
        print(f"  extractor: produced {len(all_clauses)} candidate clauses")

        # --- 4. Deduplicate ---
        all_clauses = self._deduplicate(all_clauses)
        print(f"  dedup: {len(all_clauses)} unique clauses remain")

        # --- 5. Refine: classify + risk score each clause ---
        for c in all_clauses:
            classification = self.classifier.classify(c["text"])
            c["classified_category"] = classification["category"]
            c["classifier_confidence"] = classification["confidence"]
            c["classifier_reasoning"] = classification["reasoning"]

            risk = self.risk_scorer.score(c["text"], c["classified_category"])
            c["risk_level"] = risk["risk_level"]
            c["risk_factors"] = risk["risk_factors"]
            c["risk_reasoning"] = risk["reasoning"]

        print(f"  refinement: classified and scored {len(all_clauses)} clauses")

        # --- 6. Package final output ---
        return {
            "contract_id": contract_id,
            "planned_categories": [c["name"] for c in relevant_categories],
            "n_chunks": len(chunks),
            "clauses": all_clauses,
        }

    def _plan(self, contract_text: str) -> list[dict]:
        """Ask the LLM which categories likely appear in this contract."""
        categories_block = "\n".join(
            f'- "{c["name"]}": {c["description"]}'
            for c in self.all_categories
        )
        excerpt = contract_text[:2000]

        prompt = PLANNER_PROMPT_TEMPLATE.format(
            categories_block=categories_block,
            excerpt=excerpt,
        )
        try:
            result = call_llm_json(
                prompt=prompt,
                system=PLANNER_SYSTEM,
                tag="planner",
            )
            likely_names = set(result.get("likely_categories", []))
            if not likely_names:
                # Fallback: use all categories.
                return list(self.all_categories)
            return [c for c in self.all_categories if c["name"] in likely_names]
        except Exception as e:
            print(f"  [planner] failed ({e}), using all categories")
            return list(self.all_categories)

    def _deduplicate(self, clauses: list[dict]) -> list[dict]:
        """Remove duplicate clauses (same text + same category)."""
        seen = set()
        out = []
        for c in clauses:
            key = (c.get("category", ""), c.get("text", "").strip())
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out
