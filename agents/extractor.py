"""
Extractor worker agent.

Job: given a chunk of contract text and a list of target categories,
find the exact verbatim spans of text that constitute each category's
clauses.

Key design choices:
    - The prompt asks for VERBATIM text, not paraphrases. This is
      critical for the grounding check in the validator.
    - The output is a JSON array with a strict schema.
    - "No matches" is an explicit option — the model should return
      an empty array rather than invent clauses.
    - Few-shot examples help the model understand the category
      boundaries (e.g. "Governing Law" vs "Jurisdiction").
"""

from llm_client import call_llm_json


SYSTEM_PROMPT = """You are a senior legal contract analyst with 15 years
of experience reviewing commercial contracts. You are precise, literal,
and never invent or paraphrase contract text."""


EXTRACTOR_PROMPT_TEMPLATE = """Your task: find and extract clauses from a
contract excerpt that match specific categories.

=== CATEGORIES TO LOOK FOR ===

{categories_block}

=== INSTRUCTIONS ===

1. Read the contract excerpt carefully.
2. For each category, identify one or more clauses (if any) that match.
3. For each clause you find, copy the EXACT verbatim text — do not
   paraphrase, summarise, or modify. If you shorten, use "..." to
   indicate the omission, but keep the surrounding text literal.
4. If no clause matches a category, do not include that category in the
   output. Do not fabricate clauses.
5. IMPORTANT: Do NOT extract clauses that fall outside the listed
   categories, even if they seem legally significant. Clauses about
   licensing grants, general payment terms, non-compete restrictions,
   termination for breach (as opposed to convenience), confidentiality,
   or trademark use are OUT OF SCOPE unless they specifically match
   one of the categories above.
6. Prefer precision over recall. When in doubt about whether a clause
   matches a category, DO NOT include it.
7. Assign a confidence level:
   - "high": the text clearly and unambiguously matches the category
   - "medium": the text probably matches but there is some ambiguity
   - "low": the text loosely relates to the category

=== OUTPUT FORMAT ===

Return ONLY a JSON array, no preamble. Each item has the form:

    {{
      "category": "<one of the category names above>",
      "text": "<exact verbatim text from the contract>",
      "confidence": "high" | "medium" | "low"
    }}

If no clauses are found, return [].

=== CONTRACT EXCERPT ===

{chunk_text}

=== JSON OUTPUT ===
"""


class Extractor:

    def __init__(self, categories: list[dict]):
        """
        Args:
            categories: list of {id, name, description} dicts from
                        data/categories.json.
        """
        self.categories = categories
        self._categories_block = self._format_categories(categories)

    def _format_categories(self, categories: list[dict]) -> str:
        lines = []
        for c in categories:
            lines.append(f'- "{c["name"]}": {c["description"]}')
        return "\n".join(lines)

    def extract(self, chunk_text: str) -> list[dict]:
        """
        Run extraction on a single chunk.

        Returns a list of {category, text, confidence} dicts.
        """
        prompt = EXTRACTOR_PROMPT_TEMPLATE.format(
            categories_block=self._categories_block,
            chunk_text=chunk_text,
        )
        result = call_llm_json(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            tag="extractor",
        )

        # Defensive: make sure we got a list.
        if not isinstance(result, list):
            print(f"  [extractor] expected list, got {type(result).__name__}. "
                  f"Returning [].")
            return []

        return result
