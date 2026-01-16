"""
Details Pattern Handler.

This module handles requests for detailed information about
specific frameworks.
"""

from typing import Dict, Any, Optional, List

from utils.llm import LLMClient, build_details_prompt, format_framework_details
from utils.search import SemanticSearch


def handle_details(
    framework_id: int,
    search_engine: SemanticSearch,
    llm_client: LLMClient
) -> str:
    """
    Handle request for framework details.

    Args:
        framework_id: ID of framework to show details for
        search_engine: Semantic search engine with framework data
        llm_client: LLM client for response generation

    Returns:
        Detailed framework description
    """
    result = search_engine.get_framework_by_id(framework_id)

    if not result:
        return f"I couldn't find a framework with ID {framework_id}."

    framework = result.framework_data
    return _generate_details_response(framework, llm_client)


def handle_details_by_name(
    framework_name: str,
    search_engine: SemanticSearch,
    llm_client: LLMClient
) -> str:
    """
    Handle request for framework details by name.

    Args:
        framework_name: Name of framework
        search_engine: Semantic search engine
        llm_client: LLM client

    Returns:
        Detailed framework description
    """
    result = search_engine.get_framework_by_name(framework_name)

    if not result:
        # Try fuzzy search
        fuzzy_results = search_engine.fuzzy_name_search(framework_name)
        if fuzzy_results:
            result = fuzzy_results[0]
        else:
            return f"I couldn't find a framework called '{framework_name}'. Would you like me to search for similar frameworks?"

    framework = result.framework_data
    return _generate_details_response(framework, llm_client)


def _generate_details_response(
    framework: Dict[str, Any],
    llm_client: LLMClient
) -> str:
    """
    Generate detailed response for a framework.

    Args:
        framework: Framework data dict
        llm_client: LLM client

    Returns:
        Formatted details response
    """
    system_prompt = """You are presenting detailed framework information.
Structure the information clearly with headers.
Make it scannable - consultants need quick access to key points.
End with suggested next actions."""

    user_prompt = build_details_prompt(framework)

    try:
        response = llm_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=900
        )
    except Exception:
        response = _build_fallback_details_response(framework)

    return response


def _build_fallback_details_response(framework: Dict[str, Any]) -> str:
    """
    Build fallback response when LLM is unavailable.

    Args:
        framework: Framework data

    Returns:
        Formatted details response
    """
    name = framework.get('name', 'Unknown')
    fw_type = framework.get('type', 'General')
    difficulty = framework.get('difficulty_level', 'intermediate')

    lines = [
        f"# {name}",
        "",
        f"**Type:** {fw_type}",
        f"**Difficulty:** {difficulty.capitalize()}",
        "",
        "## Overview",
        framework.get('use_case', 'No description available'),
        "",
        "## Business Domains",
        framework.get('business_domains', 'Not specified'),
        "",
        "## Problem Symptoms Addressed",
        framework.get('problem_symptoms', 'Not specified'),
        "",
    ]

    # Add inputs/outputs if available
    if framework.get('inputs_required'):
        lines.extend([
            "## Inputs Required",
            framework['inputs_required'],
            ""
        ])

    if framework.get('outputs_artifacts'):
        lines.extend([
            "## Outputs & Artifacts",
            framework['outputs_artifacts'],
            ""
        ])

    # Add diagnostic info if available
    if framework.get('diagnostic_questions'):
        lines.extend([
            "## Diagnostic Questions",
            framework['diagnostic_questions'].replace('|', '\n- '),
            ""
        ])

    if framework.get('red_flag_indicators'):
        lines.extend([
            "## Red Flags",
            framework['red_flag_indicators'],
            ""
        ])

    if framework.get('levers'):
        lines.extend([
            "## Key Levers",
            framework['levers'],
            ""
        ])

    # Related frameworks
    if framework.get('related_frameworks'):
        lines.extend([
            "## Related Frameworks",
            f"IDs: {framework['related_frameworks']}",
            ""
        ])

    # Next actions
    lines.extend([
        "---",
        "**What would you like to do next?**",
        "- Start a diagnostic session",
        "- See the framework sequence",
        "- Compare with another framework",
    ])

    return "\n".join(lines)


def format_framework_card_detailed(framework: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format framework for detailed card display.

    Args:
        framework: Framework data

    Returns:
        Formatted dict for detailed card
    """
    # Parse diagnostic questions
    questions_str = framework.get('diagnostic_questions', '')
    questions = [q.strip() for q in questions_str.split('|') if q.strip()]

    # Parse red flags
    red_flags_str = framework.get('red_flag_indicators', '')
    red_flags = [rf.strip() for rf in red_flags_str.split('|') if rf.strip()]

    # Parse levers
    levers_str = framework.get('levers', '')
    levers = [l.strip() for l in levers_str.split('|') if l.strip()]

    # Parse related frameworks
    related_str = framework.get('related_frameworks', '')
    related_ids = []
    if related_str:
        try:
            related_ids = [int(x.strip()) for x in related_str.split(',') if x.strip()]
        except ValueError:
            pass

    # Parse tags
    tags_str = framework.get('tags', '')
    tags = [t.strip() for t in tags_str.split(',') if t.strip()]

    # Parse domains
    domains_str = framework.get('business_domains', '')
    domains = [d.strip() for d in domains_str.split(',') if d.strip()]

    return {
        'id': framework.get('id'),
        'name': framework.get('name', 'Unknown'),
        'type': framework.get('type', 'General'),
        'difficulty': framework.get('difficulty_level', 'intermediate'),
        'domains': domains,
        'tags': tags,
        'problem_symptoms': framework.get('problem_symptoms', ''),
        'use_case': framework.get('use_case', ''),
        'inputs_required': framework.get('inputs_required', ''),
        'outputs_artifacts': framework.get('outputs_artifacts', ''),
        'diagnostic_questions': questions,
        'red_flags': red_flags,
        'levers': levers,
        'related_framework_ids': related_ids,
        'has_diagnostics': len(questions) > 0,
        'has_red_flags': len(red_flags) > 0,
        'has_levers': len(levers) > 0,
    }


def get_related_framework_names(
    framework: Dict[str, Any],
    search_engine: SemanticSearch
) -> List[str]:
    """
    Get names of related frameworks.

    Args:
        framework: Framework data
        search_engine: Search engine with framework data

    Returns:
        List of related framework names
    """
    related_str = framework.get('related_frameworks', '')
    if not related_str:
        return []

    names = []
    try:
        related_ids = [int(x.strip()) for x in related_str.split(',') if x.strip()]
        for rid in related_ids:
            result = search_engine.get_framework_by_id(rid)
            if result:
                names.append(result.framework_data.get('name', f'Framework #{rid}'))
    except ValueError:
        pass

    return names


def format_mini_card(framework: Dict[str, Any]) -> Dict[str, str]:
    """
    Format framework for mini/preview card display.

    Args:
        framework: Framework data

    Returns:
        Minimal dict for quick preview
    """
    return {
        'id': str(framework.get('id', '')),
        'name': framework.get('name', 'Unknown'),
        'type': framework.get('type', ''),
        'difficulty': framework.get('difficulty_level', 'intermediate'),
        'brief': framework.get('use_case', '')[:100]
    }
