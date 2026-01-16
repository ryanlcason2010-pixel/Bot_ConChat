"""
Discovery Pattern Handler.

This module handles framework discovery queries where users want to
browse, search, or list available frameworks.
"""

from typing import Dict, List, Any, Optional, Tuple

from utils.llm import LLMClient, build_discovery_prompt
from utils.search import SearchResult


def handle_discovery(
    query: str,
    search_results: List[SearchResult],
    llm_client: LLMClient,
    domains_filter: Optional[List[str]] = None,
    difficulty_filter: Optional[str] = None,
    total_count: int = 0
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Handle framework discovery queries.

    Args:
        query: User's discovery query
        search_results: List of matching frameworks from search
        llm_client: LLM client for response generation
        domains_filter: Applied domain filters
        difficulty_filter: Applied difficulty filter
        total_count: Total number of frameworks in database

    Returns:
        Tuple of (response text, list of frameworks)
    """
    if not search_results:
        response = _build_no_results_response(query, domains_filter, difficulty_filter)
        return response, []

    # Extract framework data
    frameworks = [result.framework_data for result in search_results]

    # Build response using LLM
    system_prompt = """You are a helpful assistant presenting framework options.
Be concise and organized. Group similar frameworks when possible.
Suggest ways to narrow down options if there are many results."""

    user_prompt = build_discovery_prompt(query, frameworks)

    # Add filter context
    if domains_filter or difficulty_filter:
        filter_info = []
        if domains_filter:
            filter_info.append(f"Filtered by domains: {', '.join(domains_filter)}")
        if difficulty_filter:
            filter_info.append(f"Filtered by difficulty: {difficulty_filter}")
        user_prompt += f"\n\nActive filters: {'; '.join(filter_info)}"

    try:
        response = llm_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=700
        )
    except Exception:
        response = _build_fallback_discovery_response(query, frameworks, total_count)

    return response, frameworks


def _build_no_results_response(
    query: str,
    domains_filter: Optional[List[str]],
    difficulty_filter: Optional[str]
) -> str:
    """
    Build response when no frameworks match.

    Args:
        query: User's query
        domains_filter: Applied domain filters
        difficulty_filter: Applied difficulty filter

    Returns:
        Response suggesting alternative actions
    """
    response = f"I couldn't find any frameworks matching '{query}'."

    suggestions = []
    if domains_filter:
        suggestions.append("try removing the domain filter")
    if difficulty_filter:
        suggestions.append("try selecting 'All' difficulty levels")

    if suggestions:
        response += f"\n\nYou might want to {' or '.join(suggestions)}."
    else:
        response += "\n\nTry using different keywords or describing the problem you're trying to solve."

    return response


def _build_fallback_discovery_response(
    query: str,
    frameworks: List[Dict],
    total_count: int
) -> str:
    """
    Build fallback response when LLM is unavailable.

    Args:
        query: User's query
        frameworks: Matching frameworks
        total_count: Total frameworks available

    Returns:
        Formatted discovery response
    """
    lines = [
        f"Found **{len(frameworks)}** frameworks matching your query",
        f"(out of {total_count} total frameworks):",
        ""
    ]

    # Group by type if possible
    types: Dict[str, List[Dict]] = {}
    for fw in frameworks:
        fw_type = fw.get('type', 'Other')
        if fw_type not in types:
            types[fw_type] = []
        types[fw_type].append(fw)

    for fw_type, type_frameworks in types.items():
        lines.append(f"### {fw_type}")
        for fw in type_frameworks[:3]:  # Limit per type
            name = fw.get('name', 'Unknown')
            difficulty = fw.get('difficulty_level', 'intermediate')
            use_case = fw.get('use_case', '')[:100]
            lines.append(f"- **{name}** ({difficulty}): {use_case}")
        if len(type_frameworks) > 3:
            lines.append(f"  ...and {len(type_frameworks) - 3} more")
        lines.append("")

    lines.append("Would you like to:")
    lines.append("- Get details on a specific framework")
    lines.append("- Narrow down by difficulty level")
    lines.append("- Describe a problem for recommendations")

    return "\n".join(lines)


def group_frameworks_by_type(frameworks: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group frameworks by their type.

    Args:
        frameworks: List of framework dicts

    Returns:
        Dict mapping type to list of frameworks
    """
    grouped: Dict[str, List[Dict]] = {}

    for fw in frameworks:
        fw_type = fw.get('type', 'Other')
        if fw_type not in grouped:
            grouped[fw_type] = []
        grouped[fw_type].append(fw)

    return grouped


def group_frameworks_by_domain(frameworks: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group frameworks by their primary domain.

    Args:
        frameworks: List of framework dicts

    Returns:
        Dict mapping domain to list of frameworks
    """
    grouped: Dict[str, List[Dict]] = {}

    for fw in frameworks:
        domains = fw.get('business_domains', 'Other')
        # Use first domain as primary
        primary_domain = domains.split(',')[0].strip() if domains else 'Other'

        if primary_domain not in grouped:
            grouped[primary_domain] = []
        grouped[primary_domain].append(fw)

    return grouped


def group_frameworks_by_difficulty(frameworks: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group frameworks by difficulty level.

    Args:
        frameworks: List of framework dicts

    Returns:
        Dict mapping difficulty to list of frameworks
    """
    grouped: Dict[str, List[Dict]] = {
        'beginner': [],
        'intermediate': [],
        'advanced': []
    }

    for fw in frameworks:
        difficulty = fw.get('difficulty_level', 'intermediate').lower()
        if difficulty not in grouped:
            grouped[difficulty] = []
        grouped[difficulty].append(fw)

    return grouped


def format_discovery_card(framework: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format framework for discovery card display.

    Args:
        framework: Framework data

    Returns:
        Formatted dict for card display
    """
    return {
        'id': framework.get('id'),
        'name': framework.get('name', 'Unknown'),
        'type': framework.get('type', 'General'),
        'difficulty': framework.get('difficulty_level', 'intermediate'),
        'domains': framework.get('business_domains', '').split(','),
        'brief': framework.get('use_case', '')[:150],
        'tags': framework.get('tags', '').split(','),
        'score': framework.get('score', 0)
    }


def get_discovery_suggestions(
    current_results: List[Dict],
    all_domains: List[str],
    all_types: List[str]
) -> List[str]:
    """
    Generate suggestions for refining discovery results.

    Args:
        current_results: Current search results
        all_domains: All available domains
        all_types: All available types

    Returns:
        List of suggestion strings
    """
    suggestions = []

    if len(current_results) > 10:
        suggestions.append("Filter by difficulty level to narrow results")

    if len(current_results) > 5:
        # Suggest domain filtering
        result_domains = set()
        for fw in current_results:
            domains = fw.get('business_domains', '').split(',')
            result_domains.update(d.strip() for d in domains if d.strip())

        if len(result_domains) > 1:
            suggestions.append(f"Filter by domain (found {len(result_domains)} different domains)")

    if not current_results:
        suggestions.append("Try broader search terms")
        suggestions.append("Remove active filters")

    return suggestions
