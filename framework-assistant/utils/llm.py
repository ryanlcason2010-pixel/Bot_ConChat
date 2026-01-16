"""
LLM Client Module.

This module provides the OpenAI API interface for generating responses
and embeddings, with prompt templates for each query pattern.
"""

import os
import time
from typing import Dict, Generator, List, Optional, Any

import numpy as np
from openai import OpenAI


# Prompt templates for different patterns
PROMPT_TEMPLATES = {
    'diagnostic_initial': """You are a business consultant assistant. The user described a symptom: {query}

Here are the most relevant frameworks:
{framework_list}

Respond with:
1. Confirm you understand the symptom
2. Present top 3-5 frameworks as numbered options
3. Briefly explain why each is relevant
4. Ask which framework they'd like to explore

Be concise - they're in a live meeting.""",

    'diagnostic_analysis': """Framework: {framework_name}
Diagnostic Questions: {diagnostic_questions}
User's Answers: {user_answers}
Red Flags: {red_flags}
Levers: {levers}
Related Frameworks: {related_frameworks}

Analyze the answers and provide:
- Red Flags: Which match their situation?
- Recommended Levers: Which should they focus on?
- Related Frameworks: Which might also help?
- Next Steps: 2-4 specific actions

Reference the framework's guidance directly.""",

    'discovery': """User request: {query}
Matching frameworks: {framework_list}

Present the frameworks with:
1. How many match
2. Top frameworks with brief descriptions
3. Suggest narrowing options (by difficulty, etc.)
4. Offer to deep dive or recommend starting point""",

    'details': """Framework requested: {framework_name}
Full details: {all_fields}

Present clearly:
1. Overview: What it is and when to use
2. Key Components: What makes it work
3. Prerequisites: What to do first
4. Related Frameworks: What to pair/follow with
5. Difficulty Level: Who should use this

Offer to start diagnostic, compare, or show sequence.""",

    'sequencing': """Anchor framework: {framework_name}
Related frameworks: {related_frameworks_details}

Create sequence:
- Prerequisites: Do BEFORE
- Complementary: Use ALONGSIDE
- Follow-ups: Do AFTER

Explain why this sequence matters.""",

    'comparison': """Frameworks to compare:
A: {framework_a_details}
B: {framework_b_details}

Provide:
1. Key Differences: How they differ
2. When to Use Each: Situations
3. Can They Work Together?
4. Difficulty: Which is more complex
5. Recommendation: For {scenario}, use..."""
}


class LLMClient:
    """
    OpenAI API client with retry logic and token tracking.

    Attributes:
        client: OpenAI client instance
        model: LLM model name
        embedding_model: Embedding model name
        total_prompt_tokens: Total prompt tokens used
        total_completion_tokens: Total completion tokens used
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the LLM client.

        Args:
            client: OpenAI client. If None, creates from env var.
            model: LLM model name. If None, uses env var.
            embedding_model: Embedding model name. If None, uses env var.
        """
        self.client = client or OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model or os.getenv('OPENAI_LLM_MODEL', 'gpt-4o-mini')
        self.embedding_model = embedding_model or os.getenv(
            'OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'
        )

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            system_prompt: System prompt setting context
            user_prompt: User's prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            max_retries: Maximum retry attempts

        Returns:
            Generated response text

        Raises:
            Exception: If all retries fail
        """
        max_tokens = max_tokens or int(os.getenv('LLM_MAX_TOKENS_SHORT', 500))
        temperature = temperature if temperature is not None else float(
            os.getenv('LLM_TEMPERATURE', 0.3)
        )

        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Track token usage
                if response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens
                    self.total_completion_tokens += response.usage.completion_tokens

                return response.choices[0].message.content

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = 1.0 * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)

        raise last_exception

    def generate_response_with_history(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3
    ) -> str:
        """
        Generate a response with conversation history.

        Args:
            system_prompt: System prompt setting context
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            max_retries: Maximum retry attempts

        Returns:
            Generated response text
        """
        max_tokens = max_tokens or int(os.getenv('LLM_MAX_TOKENS_LONG', 1000))
        temperature = temperature if temperature is not None else float(
            os.getenv('LLM_TEMPERATURE', 0.3)
        )

        last_exception = None

        for attempt in range(max_retries):
            try:
                all_messages = [{"role": "system", "content": system_prompt}]
                all_messages.extend(messages)

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=all_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Track token usage
                if response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens
                    self.total_completion_tokens += response.usage.completion_tokens

                return response.choices[0].message.content

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = 1.0 * (2 ** attempt)
                    time.sleep(delay)

        raise last_exception

    def generate_response_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response.

        Args:
            system_prompt: System prompt setting context
            user_prompt: User's prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Yields:
            Response chunks as they arrive
        """
        max_tokens = max_tokens or int(os.getenv('LLM_MAX_TOKENS_LONG', 1000))
        temperature = temperature if temperature is not None else float(
            os.getenv('LLM_TEMPERATURE', 0.3)
        )

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error generating response: {str(e)}"

    def embed_text(
        self,
        text: str,
        max_retries: int = 3
    ) -> np.ndarray:
        """
        Embed a text string.

        Args:
            text: Text to embed
            max_retries: Maximum retry attempts

        Returns:
            Embedding as numpy array
        """
        dimensions = int(os.getenv('EMBEDDING_DIMENSIONS', 1536))
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text,
                    dimensions=dimensions
                )

                return np.array(response.data[0].embedding)

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = 1.0 * (2 ** attempt)
                    time.sleep(delay)

        raise last_exception

    def get_token_usage(self) -> Dict[str, int]:
        """
        Get token usage statistics.

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens
        """
        return {
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_prompt_tokens + self.total_completion_tokens
        }

    def get_cost_estimate(
        self,
        input_price_per_million: float = 0.15,
        output_price_per_million: float = 0.60
    ) -> float:
        """
        Estimate cost of LLM usage.

        Args:
            input_price_per_million: Price per million input tokens
            output_price_per_million: Price per million output tokens

        Returns:
            Estimated cost in dollars
        """
        input_cost = (self.total_prompt_tokens / 1_000_000) * input_price_per_million
        output_cost = (self.total_completion_tokens / 1_000_000) * output_price_per_million
        return input_cost + output_cost

    def reset_token_usage(self) -> None:
        """Reset token usage counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0


def format_framework_list(frameworks: List[Dict[str, Any]]) -> str:
    """
    Format a list of frameworks for prompt insertion.

    Args:
        frameworks: List of framework dicts

    Returns:
        Formatted string
    """
    lines = []
    for i, fw in enumerate(frameworks, 1):
        name = fw.get('name', 'Unknown')
        domains = fw.get('business_domains', '')
        use_case = fw.get('use_case', '')
        difficulty = fw.get('difficulty_level', 'intermediate')

        lines.append(f"{i}. **{name}** ({difficulty})")
        lines.append(f"   Domains: {domains}")
        lines.append(f"   Use Case: {use_case}")
        lines.append("")

    return "\n".join(lines)


def format_framework_details(framework: Dict[str, Any]) -> str:
    """
    Format full framework details for prompt insertion.

    Args:
        framework: Framework dict

    Returns:
        Formatted string
    """
    lines = []

    # Basic info
    lines.append(f"**Name:** {framework.get('name', 'Unknown')}")
    lines.append(f"**Type:** {framework.get('type', 'Unknown')}")
    lines.append(f"**Difficulty:** {framework.get('difficulty_level', 'intermediate')}")
    lines.append("")

    # Domains and use case
    lines.append(f"**Business Domains:** {framework.get('business_domains', 'N/A')}")
    lines.append(f"**Problem Symptoms:** {framework.get('problem_symptoms', 'N/A')}")
    lines.append(f"**Use Case:** {framework.get('use_case', 'N/A')}")
    lines.append("")

    # Inputs/Outputs
    if framework.get('inputs_required'):
        lines.append(f"**Inputs Required:** {framework['inputs_required']}")
    if framework.get('outputs_artifacts'):
        lines.append(f"**Outputs/Artifacts:** {framework['outputs_artifacts']}")
    lines.append("")

    # Diagnostic info
    if framework.get('diagnostic_questions'):
        lines.append(f"**Diagnostic Questions:** {framework['diagnostic_questions']}")
    if framework.get('red_flag_indicators'):
        lines.append(f"**Red Flags:** {framework['red_flag_indicators']}")
    if framework.get('levers'):
        lines.append(f"**Levers:** {framework['levers']}")
    lines.append("")

    # Related frameworks
    if framework.get('related_frameworks'):
        lines.append(f"**Related Frameworks:** {framework['related_frameworks']}")

    return "\n".join(lines)


def build_diagnostic_initial_prompt(query: str, frameworks: List[Dict[str, Any]]) -> str:
    """
    Build prompt for initial diagnostic response.

    Args:
        query: User's symptom query
        frameworks: List of matching frameworks

    Returns:
        Formatted prompt
    """
    framework_list = format_framework_list(frameworks)
    return PROMPT_TEMPLATES['diagnostic_initial'].format(
        query=query,
        framework_list=framework_list
    )


def build_diagnostic_analysis_prompt(
    framework: Dict[str, Any],
    user_answers: str
) -> str:
    """
    Build prompt for diagnostic analysis.

    Args:
        framework: Selected framework
        user_answers: User's answers to diagnostic questions

    Returns:
        Formatted prompt
    """
    # Parse diagnostic questions (pipe-separated)
    questions = framework.get('diagnostic_questions', '').split('|')
    questions_formatted = "\n".join(f"  - {q.strip()}" for q in questions if q.strip())

    return PROMPT_TEMPLATES['diagnostic_analysis'].format(
        framework_name=framework.get('name', 'Unknown'),
        diagnostic_questions=questions_formatted,
        user_answers=user_answers,
        red_flags=framework.get('red_flag_indicators', 'None specified'),
        levers=framework.get('levers', 'None specified'),
        related_frameworks=framework.get('related_frameworks', 'None')
    )


def build_discovery_prompt(query: str, frameworks: List[Dict[str, Any]]) -> str:
    """
    Build prompt for framework discovery.

    Args:
        query: User's discovery query
        frameworks: List of matching frameworks

    Returns:
        Formatted prompt
    """
    framework_list = format_framework_list(frameworks)
    return PROMPT_TEMPLATES['discovery'].format(
        query=query,
        framework_list=framework_list
    )


def build_details_prompt(framework: Dict[str, Any]) -> str:
    """
    Build prompt for framework details.

    Args:
        framework: Framework to show details for

    Returns:
        Formatted prompt
    """
    all_fields = format_framework_details(framework)
    return PROMPT_TEMPLATES['details'].format(
        framework_name=framework.get('name', 'Unknown'),
        all_fields=all_fields
    )


def build_sequencing_prompt(
    framework: Dict[str, Any],
    related_details: List[Dict[str, Any]]
) -> str:
    """
    Build prompt for framework sequencing.

    Args:
        framework: Anchor framework
        related_details: Details of related frameworks

    Returns:
        Formatted prompt
    """
    related_formatted = "\n\n".join(
        format_framework_details(fw) for fw in related_details
    )

    return PROMPT_TEMPLATES['sequencing'].format(
        framework_name=framework.get('name', 'Unknown'),
        related_frameworks_details=related_formatted or "No related frameworks found"
    )


def build_comparison_prompt(
    framework_a: Dict[str, Any],
    framework_b: Dict[str, Any],
    scenario: str = "their situation"
) -> str:
    """
    Build prompt for framework comparison.

    Args:
        framework_a: First framework
        framework_b: Second framework
        scenario: Context for recommendation

    Returns:
        Formatted prompt
    """
    return PROMPT_TEMPLATES['comparison'].format(
        framework_a_details=format_framework_details(framework_a),
        framework_b_details=format_framework_details(framework_b),
        scenario=scenario
    )
