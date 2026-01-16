"""
Comparison Pattern Handler.

This module handles framework comparison requests, showing
side-by-side analysis of two or more frameworks.
"""

from typing import Dict, Any, List, Tuple, Optional

from utils.llm import LLMClient, build_comparison_prompt, format_framework_details
from utils.search import SemanticSearch


def handle_comparison(
    framework_ids: List[int],
    search_engine: SemanticSearch,
    llm_client: LLMClient,
    scenario: str = "their situation"
) -> str:
    """
    Handle framework comparison request.

    Args:
        framework_ids: List of framework IDs to compare (2+)
        search_engine: Semantic search engine
        llm_client: LLM client
        scenario: Context for recommendation

    Returns:
        Comparison response
    """
    if len(framework_ids) < 2:
        return "Please specify at least two frameworks to compare."

    # Get framework data
    frameworks = []
    for fid in framework_ids[:2]:  # Compare first two
        result = search_engine.get_framework_by_id(fid)
        if result:
            frameworks.append(result.framework_data)

    if len(frameworks) < 2:
        return "I couldn't find one or more of the specified frameworks."

    return _generate_comparison_response(frameworks[0], frameworks[1], llm_client, scenario)


def handle_comparison_by_names(
    name_a: str,
    name_b: str,
    search_engine: SemanticSearch,
    llm_client: LLMClient,
    scenario: str = "their situation"
) -> str:
    """
    Handle framework comparison by names.

    Args:
        name_a: First framework name
        name_b: Second framework name
        search_engine: Semantic search engine
        llm_client: LLM client
        scenario: Context for recommendation

    Returns:
        Comparison response
    """
    result_a = search_engine.get_framework_by_name(name_a)
    result_b = search_engine.get_framework_by_name(name_b)

    # Try fuzzy search if exact match fails
    if not result_a:
        fuzzy = search_engine.fuzzy_name_search(name_a)
        if fuzzy:
            result_a = fuzzy[0]

    if not result_b:
        fuzzy = search_engine.fuzzy_name_search(name_b)
        if fuzzy:
            result_b = fuzzy[0]

    if not result_a:
        return f"I couldn't find a framework called '{name_a}'."
    if not result_b:
        return f"I couldn't find a framework called '{name_b}'."

    return _generate_comparison_response(
        result_a.framework_data,
        result_b.framework_data,
        llm_client,
        scenario
    )


def _generate_comparison_response(
    framework_a: Dict[str, Any],
    framework_b: Dict[str, Any],
    llm_client: LLMClient,
    scenario: str
) -> str:
    """
    Generate comparison response for two frameworks.

    Args:
        framework_a: First framework
        framework_b: Second framework
        llm_client: LLM client
        scenario: Context for recommendation

    Returns:
        Formatted comparison response
    """
    system_prompt = """You are comparing consulting frameworks.
Provide a clear, balanced comparison.
Use a table or structured format for easy scanning.
End with a recommendation based on the scenario."""

    user_prompt = build_comparison_prompt(framework_a, framework_b, scenario)

    try:
        response = llm_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1000
        )
    except Exception:
        response = _build_fallback_comparison_response(framework_a, framework_b, scenario)

    return response


def _build_fallback_comparison_response(
    framework_a: Dict[str, Any],
    framework_b: Dict[str, Any],
    scenario: str
) -> str:
    """
    Build fallback comparison response.

    Args:
        framework_a: First framework
        framework_b: Second framework
        scenario: Context

    Returns:
        Formatted comparison
    """
    name_a = framework_a.get('name', 'Framework A')
    name_b = framework_b.get('name', 'Framework B')

    lines = [
        f"# Comparison: {name_a} vs {name_b}",
        "",
        "## Quick Overview",
        "",
        f"| Aspect | {name_a} | {name_b} |",
        "|--------|----------|----------|",
        f"| Type | {framework_a.get('type', 'N/A')} | {framework_b.get('type', 'N/A')} |",
        f"| Difficulty | {framework_a.get('difficulty_level', 'N/A')} | {framework_b.get('difficulty_level', 'N/A')} |",
        f"| Domains | {framework_a.get('business_domains', 'N/A')[:30]} | {framework_b.get('business_domains', 'N/A')[:30]} |",
        "",
        "## Use Cases",
        "",
        f"**{name_a}:**",
        framework_a.get('use_case', 'Not specified'),
        "",
        f"**{name_b}:**",
        framework_b.get('use_case', 'Not specified'),
        "",
        "## Problem Symptoms",
        "",
        f"**{name_a}:**",
        framework_a.get('problem_symptoms', 'Not specified'),
        "",
        f"**{name_b}:**",
        framework_b.get('problem_symptoms', 'Not specified'),
        "",
        "---",
        "Would you like me to:",
        "- Show detailed information for either framework",
        "- Start a diagnostic session",
        "- See the sequence for either framework",
    ]

    return "\n".join(lines)


def build_comparison_matrix(
    frameworks: List[Dict[str, Any]],
    attributes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Build a comparison matrix for multiple frameworks.

    Args:
        frameworks: List of frameworks to compare
        attributes: Attributes to compare (default if None)

    Returns:
        Matrix data structure for display
    """
    if attributes is None:
        attributes = [
            'type',
            'difficulty_level',
            'business_domains',
            'problem_symptoms',
            'use_case'
        ]

    matrix = {
        'headers': [fw.get('name', f'Framework {i}') for i, fw in enumerate(frameworks)],
        'rows': []
    }

    for attr in attributes:
        row = {
            'attribute': attr.replace('_', ' ').title(),
            'values': []
        }
        for fw in frameworks:
            value = fw.get(attr, 'N/A')
            # Truncate long values
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + '...'
            row['values'].append(value)
        matrix['rows'].append(row)

    return matrix


def calculate_framework_similarity(
    framework_a: Dict[str, Any],
    framework_b: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate similarity metrics between two frameworks.

    Args:
        framework_a: First framework
        framework_b: Second framework

    Returns:
        Dict with similarity metrics
    """
    # Domain overlap
    domains_a = set(d.strip().lower() for d in framework_a.get('business_domains', '').split(','))
    domains_b = set(d.strip().lower() for d in framework_b.get('business_domains', '').split(','))

    domain_overlap = len(domains_a & domains_b)
    domain_union = len(domains_a | domains_b)
    domain_similarity = domain_overlap / domain_union if domain_union > 0 else 0

    # Type match
    type_match = framework_a.get('type', '').lower() == framework_b.get('type', '').lower()

    # Difficulty comparison
    diff_a = framework_a.get('difficulty_level', 'intermediate').lower()
    diff_b = framework_b.get('difficulty_level', 'intermediate').lower()
    difficulty_diff = abs(_difficulty_to_num(diff_a) - _difficulty_to_num(diff_b))

    # Tag overlap
    tags_a = set(t.strip().lower() for t in framework_a.get('tags', '').split(',') if t.strip())
    tags_b = set(t.strip().lower() for t in framework_b.get('tags', '').split(',') if t.strip())

    tag_overlap = len(tags_a & tags_b)
    tag_union = len(tags_a | tags_b)
    tag_similarity = tag_overlap / tag_union if tag_union > 0 else 0

    return {
        'domain_similarity': domain_similarity,
        'domain_overlap_count': domain_overlap,
        'type_match': type_match,
        'difficulty_difference': difficulty_diff,
        'tag_similarity': tag_similarity,
        'tag_overlap_count': tag_overlap,
        'overall_similarity': (domain_similarity + tag_similarity + (1 if type_match else 0)) / 3
    }


def _difficulty_to_num(difficulty: str) -> int:
    """Convert difficulty string to number."""
    mapping = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
    return mapping.get(difficulty.lower(), 2)


def format_comparison_card(
    framework_a: Dict[str, Any],
    framework_b: Dict[str, Any],
    similarity: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format comparison data for card display.

    Args:
        framework_a: First framework
        framework_b: Second framework
        similarity: Similarity metrics

    Returns:
        Formatted dict for comparison card
    """
    return {
        'framework_a': {
            'id': framework_a.get('id'),
            'name': framework_a.get('name', 'Unknown'),
            'type': framework_a.get('type', ''),
            'difficulty': framework_a.get('difficulty_level', 'intermediate'),
            'domains': framework_a.get('business_domains', '').split(','),
            'use_case': framework_a.get('use_case', '')[:150]
        },
        'framework_b': {
            'id': framework_b.get('id'),
            'name': framework_b.get('name', 'Unknown'),
            'type': framework_b.get('type', ''),
            'difficulty': framework_b.get('difficulty_level', 'intermediate'),
            'domains': framework_b.get('business_domains', '').split(','),
            'use_case': framework_b.get('use_case', '')[:150]
        },
        'similarity': {
            'overall': f"{similarity['overall_similarity']:.0%}",
            'domain_match': f"{similarity['domain_similarity']:.0%}",
            'type_match': 'Yes' if similarity['type_match'] else 'No',
            'difficulty_gap': similarity['difficulty_difference']
        }
    }


def suggest_comparison_candidates(
    framework: Dict[str, Any],
    all_frameworks: List[Dict[str, Any]],
    max_suggestions: int = 3
) -> List[Dict[str, Any]]:
    """
    Suggest frameworks to compare with a given framework.

    Args:
        framework: Reference framework
        all_frameworks: All available frameworks
        max_suggestions: Max number of suggestions

    Returns:
        List of suggested frameworks for comparison
    """
    candidates = []

    for other in all_frameworks:
        if other.get('id') == framework.get('id'):
            continue

        similarity = calculate_framework_similarity(framework, other)

        # Good comparison candidates have some similarity but aren't identical
        if 0.3 <= similarity['overall_similarity'] <= 0.8:
            candidates.append({
                'framework': other,
                'similarity': similarity['overall_similarity']
            })

    # Sort by similarity (prefer moderate similarity)
    candidates.sort(key=lambda x: abs(x['similarity'] - 0.5))

    return [c['framework'] for c in candidates[:max_suggestions]]
