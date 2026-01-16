"""
Intent Detection Module.

This module detects user intent from natural language queries
using keyword-based rules and LLM fallback for ambiguous cases.
"""

import re
from typing import Tuple, Optional, List

from openai import OpenAI


# Intent types
INTENT_DIAGNOSTIC = "DIAGNOSTIC"
INTENT_DISCOVERY = "DISCOVERY"
INTENT_DETAILS = "DETAILS"
INTENT_SEQUENCING = "SEQUENCING"
INTENT_COMPARISON = "COMPARISON"
INTENT_UNKNOWN = "UNKNOWN"

# Problem/symptom keywords for diagnostic detection
DIAGNOSTIC_KEYWORDS = [
    # General problem indicators
    'high', 'low', 'struggling', 'facing', 'problem', 'issue', 'challenge',
    'difficulty', 'trouble', 'concern', 'worried', 'failing', 'declining',

    # Business metrics
    'turnover', 'churn', 'cac', 'ltv', 'revenue', 'profit', 'margin',
    'cost', 'expense', 'conversion', 'retention', 'acquisition', 'growth',

    # Operational issues
    'slow', 'inefficient', 'bottleneck', 'delay', 'backlog', 'overload',
    'understaffed', 'overstaffed', 'burned out', 'burnout',

    # Sales issues
    'win rate', 'close rate', 'pipeline', 'deals', 'quota', 'target',
    'stalling', 'stuck', 'losing',

    # Team/People issues
    'morale', 'engagement', 'productivity', 'performance', 'conflict',
    'communication', 'alignment', 'motivation',

    # Customer issues
    'complaints', 'satisfaction', 'nps', 'feedback', 'unhappy', 'leaving',

    # Strategic issues
    'competitive', 'market share', 'positioning', 'differentiation',
    'innovation', 'disruption', 'transformation'
]

# Discovery keywords
DISCOVERY_KEYWORDS = [
    'show me', 'list', 'what frameworks', 'find', 'search for',
    'browse', 'explore', 'all frameworks', 'available frameworks',
    'frameworks for', 'frameworks about', 'frameworks related to',
    'any frameworks', 'do you have'
]

# Details keywords
DETAILS_KEYWORDS = [
    'tell me about', 'explain', 'what is', 'describe', 'details of',
    'more about', 'deep dive', 'how does', 'overview of',
    'break down', 'walk me through', 'information on', 'info on'
]

# Sequencing keywords
SEQUENCING_KEYWORDS = [
    'before', 'after', 'prerequisites', 'what comes next', 'sequence',
    'order', 'first', 'then', 'following', 'prior to', 'preceding',
    'subsequent', 'lead into', 'precede', 'follow up', 'next steps',
    'what to do before', 'what to do after', 'preparation'
]

# Comparison keywords
COMPARISON_KEYWORDS = [
    'compare', 'vs', 'versus', 'difference between', 'differences',
    'which is better', 'pros and cons', 'advantages', 'disadvantages',
    'similarities', 'how are they different', 'compare and contrast',
    'or', 'better than'
]

# Framework name indicators (used with details detection)
FRAMEWORK_NAME_INDICATORS = [
    'the', 'framework', 'model', 'methodology', 'approach', 'technique',
    'method', 'system', 'process', 'tool'
]


def _keyword_match(query: str, keywords: List[str]) -> float:
    """
    Check if query matches any keywords.

    Args:
        query: User query (lowercased)
        keywords: List of keywords to match

    Returns:
        Match confidence (0.0-1.0)
    """
    matches = 0
    for keyword in keywords:
        if keyword in query:
            matches += 1

    if matches == 0:
        return 0.0

    # Scale confidence based on number of matches
    return min(0.9, 0.5 + (matches * 0.1))


def _detect_framework_name_mention(query: str, known_names: Optional[List[str]] = None) -> bool:
    """
    Check if query mentions a specific framework name.

    Args:
        query: User query
        known_names: Optional list of known framework names

    Returns:
        True if a framework name is detected
    """
    # Check for known names if provided
    if known_names:
        query_lower = query.lower()
        for name in known_names:
            if name.lower() in query_lower:
                return True

    # Check for capitalized words (potential framework names)
    words = query.split()
    capitalized_count = sum(
        1 for word in words
        if word[0].isupper() and len(word) > 2
        and word.lower() not in ['i', 'the', 'what', 'how', 'why', 'when', 'which']
    )

    return capitalized_count >= 1


def _detect_comparison_pattern(query: str) -> bool:
    """
    Detect if query has comparison structure (A vs B, etc.).

    Args:
        query: User query

    Returns:
        True if comparison pattern detected
    """
    # Check for "vs" or "versus"
    if ' vs ' in query.lower() or ' versus ' in query.lower():
        return True

    # Check for "compare X and Y" or "compare X to Y"
    compare_pattern = r'compare\s+\w+\s+(and|to|with)\s+\w+'
    if re.search(compare_pattern, query.lower()):
        return True

    # Check for "difference between X and Y"
    diff_pattern = r'difference\s+between\s+\w+\s+and\s+\w+'
    if re.search(diff_pattern, query.lower()):
        return True

    # Check for "X or Y" with framework indicators
    or_pattern = r'\w+\s+or\s+\w+'
    if re.search(or_pattern, query.lower()) and 'framework' in query.lower():
        return True

    return False


def detect_intent(
    query: str,
    known_framework_names: Optional[List[str]] = None,
    llm_client: Optional[OpenAI] = None
) -> Tuple[str, float]:
    """
    Detect the intent of a user query.

    Uses keyword-based detection first, then LLM fallback for ambiguous cases.

    Args:
        query: User's natural language query
        known_framework_names: Optional list of known framework names
        llm_client: Optional OpenAI client for LLM fallback

    Returns:
        Tuple of (intent_type, confidence_score)
    """
    query_lower = query.lower().strip()

    # Check for comparison pattern first (highest priority if pattern matches)
    if _detect_comparison_pattern(query):
        return INTENT_COMPARISON, 0.9

    # Score each intent type
    scores = {
        INTENT_DIAGNOSTIC: _keyword_match(query_lower, DIAGNOSTIC_KEYWORDS),
        INTENT_DISCOVERY: _keyword_match(query_lower, DISCOVERY_KEYWORDS),
        INTENT_DETAILS: _keyword_match(query_lower, DETAILS_KEYWORDS),
        INTENT_SEQUENCING: _keyword_match(query_lower, SEQUENCING_KEYWORDS),
        INTENT_COMPARISON: _keyword_match(query_lower, COMPARISON_KEYWORDS),
    }

    # Boost details score if framework name is mentioned
    if _detect_framework_name_mention(query, known_framework_names):
        scores[INTENT_DETAILS] = max(scores[INTENT_DETAILS], 0.6)

    # Find highest scoring intent
    max_score = max(scores.values())
    max_intent = max(scores.keys(), key=lambda k: scores[k])

    # If confident enough, return the result
    if max_score >= 0.6:
        return max_intent, max_score

    # If ambiguous and LLM client provided, use LLM for disambiguation
    if llm_client and max_score < 0.5:
        try:
            return _llm_intent_detection(query, llm_client)
        except Exception:
            pass  # Fall through to default handling

    # Return best guess with lower confidence
    if max_score > 0:
        return max_intent, max_score

    # Default to diagnostic for problem-like queries
    if any(word in query_lower for word in ['client', 'customer', 'team', 'company', 'business']):
        return INTENT_DIAGNOSTIC, 0.4

    return INTENT_UNKNOWN, 0.0


def _llm_intent_detection(query: str, client: OpenAI) -> Tuple[str, float]:
    """
    Use LLM to detect intent for ambiguous queries.

    Args:
        query: User query
        client: OpenAI client

    Returns:
        Tuple of (intent_type, confidence)
    """
    system_prompt = """You are an intent classifier for a consulting framework assistant.
Classify the user's query into exactly one of these categories:
- DIAGNOSTIC: User describes a problem/symptom and needs framework recommendations
- DISCOVERY: User wants to browse/search/list available frameworks
- DETAILS: User wants information about a specific framework
- SEQUENCING: User asks about framework order/prerequisites/next steps
- COMPARISON: User wants to compare two or more frameworks

Respond with ONLY the category name and confidence (0-1), separated by comma.
Example: DIAGNOSTIC, 0.85"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=20,
        temperature=0.1
    )

    result = response.choices[0].message.content.strip()

    # Parse response
    parts = result.split(',')
    intent = parts[0].strip().upper()
    confidence = float(parts[1].strip()) if len(parts) > 1 else 0.7

    # Validate intent
    valid_intents = [
        INTENT_DIAGNOSTIC, INTENT_DISCOVERY, INTENT_DETAILS,
        INTENT_SEQUENCING, INTENT_COMPARISON
    ]

    if intent not in valid_intents:
        return INTENT_UNKNOWN, 0.0

    return intent, confidence


def extract_framework_names(query: str, known_names: List[str]) -> List[str]:
    """
    Extract framework names mentioned in a query.

    Args:
        query: User query
        known_names: List of known framework names

    Returns:
        List of mentioned framework names
    """
    found = []
    query_lower = query.lower()

    for name in known_names:
        if name.lower() in query_lower:
            found.append(name)

    return found


def extract_comparison_frameworks(query: str, known_names: List[str]) -> Tuple[str, str]:
    """
    Extract two framework names for comparison.

    Args:
        query: User query
        known_names: List of known framework names

    Returns:
        Tuple of (framework_a, framework_b) or empty strings if not found
    """
    found = extract_framework_names(query, known_names)

    if len(found) >= 2:
        return found[0], found[1]
    elif len(found) == 1:
        return found[0], ""
    else:
        return "", ""


def get_intent_description(intent: str) -> str:
    """
    Get a human-readable description of an intent.

    Args:
        intent: Intent type

    Returns:
        Description string
    """
    descriptions = {
        INTENT_DIAGNOSTIC: "Analyzing symptoms to recommend frameworks",
        INTENT_DISCOVERY: "Browsing and searching frameworks",
        INTENT_DETAILS: "Getting detailed framework information",
        INTENT_SEQUENCING: "Understanding framework order and prerequisites",
        INTENT_COMPARISON: "Comparing frameworks",
        INTENT_UNKNOWN: "General inquiry"
    }
    return descriptions.get(intent, "Unknown intent")
