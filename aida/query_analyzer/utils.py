"""
JSON Response Parser

Utility for extracting JSON from LLM responses.
Handles various formats: raw JSON, markdown code blocks, embedded JSON.
"""

import json
import re
from typing import Dict, Any, Optional


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from LLM response.

    Attempts multiple parsing strategies:
    1. Direct JSON parse
    2. Extract from markdown code block (```json ... ```)
    3. Find embedded JSON object in text

    Handles inline comments (// style) by stripping them before parsing.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON dict, or None if parsing fails
    """
    def strip_comments(text: str) -> str:
        """Remove inline // comments from JSON text."""
        return re.sub(r'//.*?(?=\n|$)', '', text, flags=re.MULTILINE)

    # Strategy 1: Direct parse
    try:
        cleaned = strip_comments(response.strip())
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if match:
        try:
            cleaned = strip_comments(match.group(1))
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find JSON object (handles nested braces)
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if match:
        try:
            cleaned = strip_comments(match.group(0))
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    return None
