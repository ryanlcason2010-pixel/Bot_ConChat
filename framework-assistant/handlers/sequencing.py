"""
Sequencing Pattern Handler.

This module handles requests about framework sequencing,
prerequisites, and follow-up frameworks.
"""

from typing import Dict, Any, List, Tuple, Optional

from utils.llm import LLMClient, build_sequencing_prompt
from utils.search import SemanticSearch


def handle_sequencing(
    framework_id: int,
    search_engine: SemanticSearch,
    llm_client: LLMClient
) -> str:
    """
    Handle framework sequencing request.

    Args:
        framework_id: ID of anchor framework
        search_engine: Semantic search engine
        llm_client: LLM client

    Returns:
        Sequencing response with prerequisites, complementary, and follow-ups
    """
    result = search_engine.get_framework_by_id(framework_id)

    if not result:
        return f"I couldn't find a framework with ID {framework_id}."

    framework = result.framework_data
    return _generate_sequencing_response(framework, search_engine, llm_client)


def handle_sequencing_by_name(
    framework_name: str,
    search_engine: SemanticSearch,
    llm_client: LLMClient
) -> str:
    """
    Handle framework sequencing request by name.

    Args:
        framework_name: Name of anchor framework
        search_engine: Semantic search engine
        llm_client: LLM client

    Returns:
        Sequencing response
    """
    result = search_engine.get_framework_by_name(framework_name)

    if not result:
        fuzzy_results = search_engine.fuzzy_name_search(framework_name)
        if fuzzy_results:
            result = fuzzy_results[0]
        else:
            return f"I couldn't find a framework called '{framework_name}'."

    framework = result.framework_data
    return _generate_sequencing_response(framework, search_engine, llm_client)


def _generate_sequencing_response(
    framework: Dict[str, Any],
    search_engine: SemanticSearch,
    llm_client: LLMClient
) -> str:
    """
    Generate sequencing response for a framework.

    Args:
        framework: Framework data
        search_engine: Semantic search engine
        llm_client: LLM client

    Returns:
        Formatted sequencing response
    """
    # Get related frameworks
    related_details = get_related_framework_details(framework, search_engine)

    # If no related frameworks defined, find semantically similar ones
    if not related_details:
        similar = search_engine.get_related_frameworks(framework.get('id', 0), top_k=5)
        related_details = [r.framework_data for r in similar]

    # Generate response
    system_prompt = """You are helping a consultant understand framework sequencing.
Clearly categorize frameworks as:
- Prerequisites (do BEFORE the main framework)
- Complementary (use ALONGSIDE)
- Follow-ups (do AFTER)

Be practical about why the sequence matters."""

    user_prompt = build_sequencing_prompt(framework, related_details)

    try:
        response = llm_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=800
        )
    except Exception:
        response = _build_fallback_sequencing_response(framework, related_details)

    return response


def _build_fallback_sequencing_response(
    framework: Dict[str, Any],
    related: List[Dict[str, Any]]
) -> str:
    """
    Build fallback sequencing response.

    Args:
        framework: Anchor framework
        related: Related frameworks

    Returns:
        Formatted response
    """
    name = framework.get('name', 'Unknown')

    lines = [
        f"# Framework Sequence for {name}",
        "",
    ]

    if related:
        lines.append("## Related Frameworks")
        lines.append("")

        for rf in related[:5]:
            rf_name = rf.get('name', 'Unknown')
            rf_use = rf.get('use_case', '')[:100]
            lines.append(f"- **{rf_name}**: {rf_use}")

        lines.extend([
            "",
            "*Note: Without explicit sequencing data, consider the difficulty levels and prerequisites when ordering these frameworks.*",
            ""
        ])
    else:
        lines.extend([
            "No explicitly related frameworks are defined for this framework.",
            "",
            "Consider exploring semantically similar frameworks or describing your workflow for recommendations.",
        ])

    lines.extend([
        "",
        "---",
        "Would you like me to:",
        "- Compare any of these frameworks",
        "- Show details for a specific framework",
        "- Search for frameworks by a specific criteria",
    ])

    return "\n".join(lines)


def get_related_framework_details(
    framework: Dict[str, Any],
    search_engine: SemanticSearch
) -> List[Dict[str, Any]]:
    """
    Get full details for related frameworks.

    Args:
        framework: Framework with related_frameworks field
        search_engine: Semantic search engine

    Returns:
        List of related framework data dicts
    """
    related_str = framework.get('related_frameworks', '')
    if not related_str:
        return []

    details = []
    try:
        related_ids = [int(x.strip()) for x in related_str.split(',') if x.strip()]
        for rid in related_ids:
            result = search_engine.get_framework_by_id(rid)
            if result:
                details.append(result.framework_data)
    except ValueError:
        pass

    return details


def categorize_related_frameworks(
    anchor: Dict[str, Any],
    related: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize related frameworks by relationship type.

    Uses difficulty level and domain overlap as heuristics.

    Args:
        anchor: Anchor framework
        related: List of related frameworks

    Returns:
        Dict with 'prerequisites', 'complementary', 'follow_ups' keys
    """
    anchor_difficulty = _difficulty_to_num(anchor.get('difficulty_level', 'intermediate'))
    anchor_domains = set(d.strip().lower() for d in anchor.get('business_domains', '').split(','))

    categories = {
        'prerequisites': [],
        'complementary': [],
        'follow_ups': []
    }

    for rf in related:
        rf_difficulty = _difficulty_to_num(rf.get('difficulty_level', 'intermediate'))
        rf_domains = set(d.strip().lower() for d in rf.get('business_domains', '').split(','))

        # Domain overlap
        overlap = len(anchor_domains & rf_domains)

        # Heuristic categorization
        if rf_difficulty < anchor_difficulty:
            # Simpler frameworks are likely prerequisites
            categories['prerequisites'].append(rf)
        elif rf_difficulty > anchor_difficulty:
            # More complex frameworks are likely follow-ups
            categories['follow_ups'].append(rf)
        elif overlap > 0:
            # Same difficulty with domain overlap = complementary
            categories['complementary'].append(rf)
        else:
            # Default to follow-ups
            categories['follow_ups'].append(rf)

    return categories


def _difficulty_to_num(difficulty: str) -> int:
    """Convert difficulty string to number for comparison."""
    mapping = {
        'beginner': 1,
        'intermediate': 2,
        'advanced': 3
    }
    return mapping.get(difficulty.lower(), 2)


def format_sequence_card(
    anchor: Dict[str, Any],
    categories: Dict[str, List[Dict]]
) -> Dict[str, Any]:
    """
    Format sequence information for display.

    Args:
        anchor: Anchor framework
        categories: Categorized related frameworks

    Returns:
        Formatted dict for sequence display
    """
    def format_framework_mini(fw: Dict) -> Dict:
        return {
            'id': fw.get('id'),
            'name': fw.get('name', 'Unknown'),
            'difficulty': fw.get('difficulty_level', 'intermediate'),
            'brief': fw.get('use_case', '')[:80]
        }

    return {
        'anchor': {
            'id': anchor.get('id'),
            'name': anchor.get('name', 'Unknown'),
            'difficulty': anchor.get('difficulty_level', 'intermediate')
        },
        'prerequisites': [format_framework_mini(f) for f in categories.get('prerequisites', [])],
        'complementary': [format_framework_mini(f) for f in categories.get('complementary', [])],
        'follow_ups': [format_framework_mini(f) for f in categories.get('follow_ups', [])],
        'total_related': sum(len(v) for v in categories.values())
    }


def build_learning_path(
    start_framework_id: int,
    search_engine: SemanticSearch,
    max_depth: int = 3
) -> List[Dict[str, Any]]:
    """
    Build a learning path starting from a framework.

    Args:
        start_framework_id: Starting framework ID
        search_engine: Semantic search engine
        max_depth: Maximum path depth

    Returns:
        Ordered list of frameworks in suggested learning order
    """
    path = []
    seen = set()

    def add_to_path(fid: int, depth: int):
        if depth > max_depth or fid in seen:
            return

        seen.add(fid)
        result = search_engine.get_framework_by_id(fid)
        if not result:
            return

        fw = result.framework_data
        path.append(fw)

        # Recursively add related frameworks
        related_str = fw.get('related_frameworks', '')
        if related_str:
            try:
                related_ids = [int(x.strip()) for x in related_str.split(',') if x.strip()]
                for rid in related_ids:
                    add_to_path(rid, depth + 1)
            except ValueError:
                pass

    add_to_path(start_framework_id, 0)

    # Sort by difficulty for learning path
    path.sort(key=lambda x: _difficulty_to_num(x.get('difficulty_level', 'intermediate')))

    return path
