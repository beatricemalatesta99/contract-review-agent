"""
Segmenter worker agent.

Job: take a raw contract string and break it into reasonable chunks.

Why segment at all?
    - Long contracts (50+ pages) are fine for Claude's context window,
      but shorter chunks give more focused extraction and cheaper calls.
    - Contracts have natural boundaries (sections, articles) — we
      exploit them with a simple heuristic before falling back to LLM.

We try a deterministic split first (cheap, reliable) and only fall back
to LLM segmentation if the rule-based split produces nothing sensible.
This is a small example of the "LLM + rules" philosophy that's the
design principle of the whole system.
"""

import re


class Segmenter:

    # Common section-header patterns in CUAD-style contracts.
    SECTION_PATTERNS = [
        r"\n\s*(?:ARTICLE|Article|SECTION|Section)\s+\d+[\.\:]?",
        r"\n\s*\d+\.\s+[A-Z][A-Z\s]{3,}\n",   # "1. TERMS AND CONDITIONS"
        r"\n\s*\d+\.\d+\s+",                   # "1.1 Scope"
    ]

    # Target maximum characters per chunk. ~15k chars ≈ ~4k tokens,
    # which keeps individual worker calls cheap and focused.
    MAX_CHARS_PER_CHUNK = 30000

    def split(self, contract_text: str) -> list[dict]:
        """
        Split a contract into chunks.

        Returns a list of {"chunk_id": int, "text": str} dicts.
        """
        # 1. Try rule-based split on section headers.
        chunks = self._rule_based_split(contract_text)

        # 2. If chunks are too big, split them further by paragraphs.
        chunks = self._ensure_max_size(chunks)

        # 3. Return with chunk IDs.
        return [{"chunk_id": i, "text": c} for i, c in enumerate(chunks)]

    def _rule_based_split(self, text: str) -> list[str]:
        """Split on section headers. Returns a list of text chunks."""
        # Combine patterns with alternation.
        pattern = "|".join(self.SECTION_PATTERNS)
        # Find all section start positions.
        positions = [m.start() for m in re.finditer(pattern, text)]

        # If no sections detected, return the whole text as one chunk.
        if not positions:
            return [text]

        # Add 0 and len(text) as boundaries.
        positions = [0] + positions + [len(text)]
        chunks = []
        for i in range(len(positions) - 1):
            chunk = text[positions[i]:positions[i+1]].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _ensure_max_size(self, chunks: list[str]) -> list[str]:
        """If any chunk is too big, split it on paragraph breaks."""
        out = []
        for chunk in chunks:
            if len(chunk) <= self.MAX_CHARS_PER_CHUNK:
                out.append(chunk)
                continue

            # Split by paragraph boundaries, accumulate until the limit.
            paragraphs = chunk.split("\n\n")
            buffer = ""
            for p in paragraphs:
                if len(buffer) + len(p) + 2 > self.MAX_CHARS_PER_CHUNK and buffer:
                    out.append(buffer.strip())
                    buffer = p
                else:
                    buffer = buffer + "\n\n" + p if buffer else p
            if buffer.strip():
                out.append(buffer.strip())
        return out
