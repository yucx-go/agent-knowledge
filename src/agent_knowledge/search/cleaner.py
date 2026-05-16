"""MemoryCleaner: preprocess text for higher-quality indexing.

Strips markdown formatting, extracts entities from backticks,
normalizes whitespace — so BM25/exact search sees clean semantic content.

Usage example::

    from agent_knowledge.search.cleaner import MemoryCleaner

    raw_md = "## Title\n**bold** text with `code_ref` and [link](url)"

    # Clean markdown → plain text
    plain = MemoryCleaner.clean(raw_md)
    # → "Title\nbold text with code_ref and link"

    # Extract technical entities before cleaning
    entities = MemoryCleaner.extract_entities(raw_md)
    # → ["code_ref"]

    # One-pass: clean + extract
    cleaned, entities = MemoryCleaner.clean_for_index(raw_md)
"""

from __future__ import annotations

import re


class MemoryCleaner:
    """Clean markdown text into pure-text facts for better search indexing."""

    # Markdown patterns to strip
    _MD_PATTERNS = [
        # Fenced code blocks (```...```) — remove the fences, keep content
        (re.compile(r'```[\w]*\n(.*?)```', re.DOTALL), r'\1'),
        # Inline code backticks — unwrap content
        (re.compile(r'`([^`]+)`'), r'\1'),
        # Bold **text** or __text__
        (re.compile(r'\*\*(.+?)\*\*'), r'\1'),
        (re.compile(r'__(.+?)__'), r'\1'),
        # Italic *text* or _text_ (but not snake_case)
        (re.compile(r'(?<!\w)\*(.+?)\*(?!\w)'), r'\1'),
        # Strikethrough ~~text~~
        (re.compile(r'~~(.+?)~~'), r'\1'),
        # Headers: ## Title → Title
        (re.compile(r'^#{1,6}\s+', re.MULTILINE), ''),
        # Links: [text](url) → text
        (re.compile(r'\[([^\]]+)\]\([^)]+\)'), r'\1'),
        # Images: ![alt](url) → alt
        (re.compile(r'!\[([^\]]*)\]\([^)]+\)'), r'\1'),
        # HTML tags
        (re.compile(r'<[^>]+>'), ''),
        # Horizontal rules
        (re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE), ''),
        # Blockquotes: > text → text
        (re.compile(r'^>\s?', re.MULTILINE), ''),
        # Table separators: |---|---|
        (re.compile(r'^\|[-:| ]+\|\s*$', re.MULTILINE), ''),
        # Leading pipe in table rows: | cell | cell | → cell  cell
        (re.compile(r'^\||\|$', re.MULTILINE), ''),
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        """Clean markdown text to plain text for indexing.

        Preserves semantic content while removing formatting noise.
        """
        result = text

        for pattern, replacement in cls._MD_PATTERNS:
            result = pattern.sub(replacement, result)

        # Normalize whitespace: collapse multiple spaces/newlines
        result = re.sub(r'[ \t]+', ' ', result)
        result = re.sub(r'\n{3,}', '\n\n', result)

        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in result.split('\n')]
        result = '\n'.join(lines)

        return result.strip()

    @classmethod
    def extract_entities(cls, text: str) -> list[str]:
        """Extract notable entities from text.

        Sources:
        - Backtick-wrapped identifiers: `collapsible_panel` → collapsible_panel
        - Technical identifiers: snake_case, IDs with prefixes
        - Quoted terms
        """
        entities: list[str] = []
        seen = set()

        # Backtick-wrapped entities
        for m in re.finditer(r'`([^`]+)`', text):
            entity = m.group(1).strip()
            if entity and entity.lower() not in seen and len(entity) < 100:
                seen.add(entity.lower())
                entities.append(entity)

        # Technical identifiers (snake_case, prefixed IDs)
        for m in re.finditer(r'\b([a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+)\b', text):
            entity = m.group(1)
            if entity.lower() not in seen:
                seen.add(entity.lower())
                entities.append(entity)

        # Prefixed IDs (tbl_xxx, ou_xxx, etc.)
        for m in re.finditer(r'\b((?:tbl|ou_|oc_|om_|img_|file_)[a-zA-Z0-9]+)\b', text):
            entity = m.group(1)
            if entity.lower() not in seen:
                seen.add(entity.lower())
                entities.append(entity)

        return entities

    @classmethod
    def clean_for_index(cls, text: str) -> tuple[str, list[str]]:
        """Clean text and extract entities in one pass.

        Returns (cleaned_text, entities).
        Entities are extracted BEFORE cleaning (from the original markdown).
        """
        entities = cls.extract_entities(text)
        cleaned = cls.clean(text)

        # Append extracted entities as a searchable "entities" section
        # so BM25 can match them even after markdown stripping
        if entities:
            entity_line = " ".join(entities)
            cleaned = cleaned + "\n" + entity_line

        return cleaned, entities
