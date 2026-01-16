"""
Diagnostic Pattern Handler.

This module handles the diagnostic flow: symptom description → framework
selection → diagnostic questions → analysis and recommendations.
"""

from typing import Dict, List, Any, Optional, Tuple

from utils.llm import (
    LLMClient,
    build_diagnostic_initial_prompt,
    build_diagnostic_analysis_prompt,
    format_framework_list
)
from utils.search import SearchResult
from utils.session import SessionManager


def handle_diagnostic(
    query: str,
    search_results: List[SearchResult],
    session: SessionManager,
    llm_client: LLMClient
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Handle the initial diagnostic query.

    Presents matching frameworks and asks user to select one.

    Args:
        query: User's symptom description
        search_results: List of matching frameworks
        session: Session manager
        llm_client: LLM client for response generation

    Returns:
        Tuple of (response text, list of frameworks shown)
    """
    if not search_results:
        return (
            "I couldn't find any frameworks that match your description. "
            "Could you provide more details about the problem you're facing?",
            []
        )

    # Prepare framework data for prompt
    frameworks = [result.framework_data for result in search_results[:5]]

    # Generate response using LLM
    system_prompt = """You are a helpful business consultant assistant.
Be concise and practical - the user may be in a live client meeting."""

    user_prompt = build_diagnostic_initial_prompt(query, frameworks)

    try:
        response = llm_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=800
        )
    except Exception as e:
        # Fallback response if LLM fails
        response = _build_fallback_diagnostic_response(query, frameworks)

    # Update session state
    session.set_current_stage('framework_selection')
    session.set_last_query(query)

    # Record viewed frameworks
    for fw in frameworks:
        session.add_framework_viewed(fw.get('name', 'Unknown'), fw.get('id', 0))

    return response, frameworks


def _build_fallback_diagnostic_response(query: str, frameworks: List[Dict]) -> str:
    """
    Build a fallback response when LLM is unavailable.

    Args:
        query: User's query
        frameworks: List of matching frameworks

    Returns:
        Formatted response string
    """
    lines = [
        f"Based on your description about '{query[:100]}...', here are relevant frameworks:",
        ""
    ]

    for i, fw in enumerate(frameworks[:5], 1):
        name = fw.get('name', 'Unknown')
        use_case = fw.get('use_case', '')[:150]
        difficulty = fw.get('difficulty_level', 'intermediate')

        lines.append(f"**{i}. {name}** ({difficulty})")
        lines.append(f"   {use_case}")
        lines.append("")

    lines.append("Which framework would you like to explore? (Enter the number or name)")

    return "\n".join(lines)


def handle_framework_selection(
    selection: str,
    frameworks: List[Dict[str, Any]],
    session: SessionManager
) -> Tuple[Optional[Dict], str]:
    """
    Handle user's framework selection.

    Args:
        selection: User's selection (number or name)
        frameworks: List of available frameworks
        session: Session manager

    Returns:
        Tuple of (selected framework dict or None, message)
    """
    selected = None

    # Try to parse as number
    try:
        index = int(selection.strip()) - 1
        if 0 <= index < len(frameworks):
            selected = frameworks[index]
    except ValueError:
        pass

    # Try to match by name
    if not selected:
        selection_lower = selection.lower().strip()
        for fw in frameworks:
            if fw.get('name', '').lower() in selection_lower or selection_lower in fw.get('name', '').lower():
                selected = fw
                break

    if selected:
        session.set_selected_framework(selected)
        session.set_current_stage('diagnostic_active')

        # Format diagnostic questions
        questions = selected.get('diagnostic_questions', '')
        if questions:
            question_list = [q.strip() for q in questions.split('|') if q.strip()]
            msg = f"Great choice! Let's diagnose using **{selected['name']}**.\n\n"
            msg += "Please answer these diagnostic questions:\n\n"
            for i, q in enumerate(question_list, 1):
                msg += f"{i}. {q}\n"
            msg += "\nYou can answer all at once or one by one."
            return selected, msg
        else:
            msg = f"**{selected['name']}** selected. This framework doesn't have specific diagnostic questions. "
            msg += "Would you like me to explain how to apply it to your situation?"
            return selected, msg
    else:
        return None, "I didn't catch that. Please enter the number or name of the framework you'd like to explore."


def handle_diagnostic_analysis(
    user_answers: str,
    framework: Dict[str, Any],
    session: SessionManager,
    llm_client: LLMClient
) -> str:
    """
    Analyze diagnostic answers and provide recommendations.

    Args:
        user_answers: User's answers to diagnostic questions
        framework: Selected framework
        session: Session manager
        llm_client: LLM client

    Returns:
        Analysis response with red flags, levers, and next steps
    """
    # Store answers
    session.add_diagnostic_answer('combined_answers', user_answers)

    # Build analysis prompt
    system_prompt = """You are an expert business consultant providing diagnostic analysis.
Be specific and actionable. Reference the framework's guidance directly.
Format your response with clear sections using markdown."""

    user_prompt = build_diagnostic_analysis_prompt(framework, user_answers)

    try:
        response = llm_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1000
        )
    except Exception as e:
        response = _build_fallback_analysis(framework, user_answers)

    # Update stage
    session.set_current_stage('diagnostic_complete')

    return response


def _build_fallback_analysis(framework: Dict, answers: str) -> str:
    """
    Build fallback analysis when LLM is unavailable.

    Args:
        framework: Framework data
        answers: User's answers

    Returns:
        Basic analysis response
    """
    name = framework.get('name', 'the framework')
    red_flags = framework.get('red_flag_indicators', 'None specified')
    levers = framework.get('levers', 'None specified')
    related = framework.get('related_frameworks', 'None')

    return f"""## Analysis for {name}

Based on your answers, here are key considerations:

### Red Flags to Watch For
{red_flags}

### Recommended Levers
{levers}

### Related Frameworks
{related}

### Next Steps
1. Review the red flags against your specific situation
2. Prioritize 2-3 levers to focus on first
3. Consider exploring related frameworks for additional support

Would you like me to explain any of these in more detail?"""


def get_diagnostic_questions(framework: Dict[str, Any]) -> List[str]:
    """
    Extract diagnostic questions from a framework.

    Args:
        framework: Framework data dict

    Returns:
        List of diagnostic questions
    """
    questions_str = framework.get('diagnostic_questions', '')
    if not questions_str:
        return []

    return [q.strip() for q in questions_str.split('|') if q.strip()]


def format_diagnostic_card(framework: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format framework data for diagnostic card display.

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
        'domains': framework.get('business_domains', ''),
        'symptoms': framework.get('problem_symptoms', ''),
        'use_case': framework.get('use_case', ''),
        'diagnostic_questions': get_diagnostic_questions(framework),
        'red_flags': framework.get('red_flag_indicators', ''),
        'levers': framework.get('levers', ''),
        'has_diagnostics': bool(framework.get('diagnostic_questions'))
    }
