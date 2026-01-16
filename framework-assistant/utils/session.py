"""
Session Management Module.

This module handles conversation session state, including
message history, viewed frameworks, and multi-turn flow state.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st


class SessionManager:
    """
    Manages session state for the Framework Assistant.

    Uses Streamlit's session_state for persistence across reruns.

    Attributes:
        session_key_prefix: Prefix for session state keys
    """

    def __init__(self, session_key_prefix: str = "fa_"):
        """
        Initialize the session manager.

        Args:
            session_key_prefix: Prefix for session state keys
        """
        self.session_key_prefix = session_key_prefix
        self._initialize_session()

    def _get_key(self, name: str) -> str:
        """Get full session state key."""
        return f"{self.session_key_prefix}{name}"

    def _initialize_session(self) -> None:
        """Initialize session state with default values."""
        defaults = {
            'conversation_history': [],
            'frameworks_viewed': [],
            'last_query': '',
            'current_stage': 'initial',  # initial, framework_selected, diagnostic_active
            'selected_framework': None,
            'selected_framework_id': None,
            'diagnostic_answers': {},
            'comparison_frameworks': [],
            'session_start': datetime.now().isoformat(),
            'query_count': 0,
            'feedback_given': [],
        }

        for key, default in defaults.items():
            full_key = self._get_key(key)
            if full_key not in st.session_state:
                st.session_state[full_key] = default

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a message to conversation history.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata (intent, frameworks shown, etc.)
        """
        key = self._get_key('conversation_history')
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        st.session_state[key].append(message)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the full conversation history.

        Returns:
            List of message dicts
        """
        return st.session_state[self._get_key('conversation_history')]

    def get_messages_for_llm(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """
        Get recent messages formatted for LLM API.

        Args:
            max_messages: Maximum number of messages to include

        Returns:
            List of message dicts with 'role' and 'content'
        """
        history = self.get_conversation_history()
        recent = history[-max_messages:] if len(history) > max_messages else history

        return [
            {'role': msg['role'], 'content': msg['content']}
            for msg in recent
        ]

    def get_conversation_context(self, include_frameworks: bool = True) -> str:
        """
        Get conversation context as a formatted string.

        Args:
            include_frameworks: Whether to include viewed frameworks

        Returns:
            Formatted context string
        """
        lines = []

        # Recent messages
        history = self.get_conversation_history()
        for msg in history[-5:]:
            role = msg['role'].capitalize()
            content = msg['content'][:200] + '...' if len(msg['content']) > 200 else msg['content']
            lines.append(f"{role}: {content}")

        # Viewed frameworks
        if include_frameworks:
            viewed = self.get_frameworks_viewed()
            if viewed:
                lines.append(f"\nPreviously discussed frameworks: {', '.join(viewed)}")

        return "\n".join(lines)

    def add_framework_viewed(self, framework_name: str, framework_id: int) -> None:
        """
        Record that a framework was viewed/discussed.

        Args:
            framework_name: Name of the framework
            framework_id: ID of the framework
        """
        key = self._get_key('frameworks_viewed')
        entry = {'name': framework_name, 'id': framework_id, 'time': datetime.now().isoformat()}

        # Avoid duplicates
        existing = st.session_state[key]
        if not any(f['id'] == framework_id for f in existing):
            st.session_state[key].append(entry)

    def get_frameworks_viewed(self) -> List[str]:
        """
        Get list of viewed framework names.

        Returns:
            List of framework names
        """
        return [f['name'] for f in st.session_state[self._get_key('frameworks_viewed')]]

    def get_frameworks_viewed_ids(self) -> List[int]:
        """
        Get list of viewed framework IDs.

        Returns:
            List of framework IDs
        """
        return [f['id'] for f in st.session_state[self._get_key('frameworks_viewed')]]

    def set_last_query(self, query: str) -> None:
        """
        Set the last query.

        Args:
            query: User's query
        """
        st.session_state[self._get_key('last_query')] = query
        st.session_state[self._get_key('query_count')] += 1

    def get_last_query(self) -> str:
        """
        Get the last query.

        Returns:
            Last query string
        """
        return st.session_state[self._get_key('last_query')]

    def get_query_count(self) -> int:
        """
        Get total query count.

        Returns:
            Number of queries in session
        """
        return st.session_state[self._get_key('query_count')]

    def set_current_stage(self, stage: str) -> None:
        """
        Set the current conversation stage.

        Args:
            stage: Stage name (initial, framework_selected, diagnostic_active)
        """
        st.session_state[self._get_key('current_stage')] = stage

    def get_current_stage(self) -> str:
        """
        Get the current conversation stage.

        Returns:
            Current stage name
        """
        return st.session_state[self._get_key('current_stage')]

    def set_selected_framework(self, framework: Optional[Dict]) -> None:
        """
        Set the currently selected framework.

        Args:
            framework: Framework dict or None
        """
        st.session_state[self._get_key('selected_framework')] = framework
        if framework:
            st.session_state[self._get_key('selected_framework_id')] = framework.get('id')
        else:
            st.session_state[self._get_key('selected_framework_id')] = None

    def get_selected_framework(self) -> Optional[Dict]:
        """
        Get the currently selected framework.

        Returns:
            Framework dict or None
        """
        return st.session_state[self._get_key('selected_framework')]

    def get_selected_framework_id(self) -> Optional[int]:
        """
        Get the ID of the currently selected framework.

        Returns:
            Framework ID or None
        """
        return st.session_state[self._get_key('selected_framework_id')]

    def set_diagnostic_answers(self, answers: Dict[str, str]) -> None:
        """
        Set diagnostic answers.

        Args:
            answers: Dict mapping question to answer
        """
        st.session_state[self._get_key('diagnostic_answers')] = answers

    def add_diagnostic_answer(self, question: str, answer: str) -> None:
        """
        Add a single diagnostic answer.

        Args:
            question: The diagnostic question
            answer: User's answer
        """
        key = self._get_key('diagnostic_answers')
        st.session_state[key][question] = answer

    def get_diagnostic_answers(self) -> Dict[str, str]:
        """
        Get all diagnostic answers.

        Returns:
            Dict mapping questions to answers
        """
        return st.session_state[self._get_key('diagnostic_answers')]

    def clear_diagnostic_answers(self) -> None:
        """Clear all diagnostic answers."""
        st.session_state[self._get_key('diagnostic_answers')] = {}

    def set_comparison_frameworks(self, framework_ids: List[int]) -> None:
        """
        Set frameworks for comparison.

        Args:
            framework_ids: List of framework IDs to compare
        """
        st.session_state[self._get_key('comparison_frameworks')] = framework_ids

    def get_comparison_frameworks(self) -> List[int]:
        """
        Get frameworks for comparison.

        Returns:
            List of framework IDs
        """
        return st.session_state[self._get_key('comparison_frameworks')]

    def add_feedback(self, framework_id: int, rating: int, query: str) -> None:
        """
        Record feedback for a framework.

        Args:
            framework_id: ID of the framework
            rating: 1 for thumbs up, -1 for thumbs down
            query: The query that led to this framework
        """
        key = self._get_key('feedback_given')
        entry = {
            'framework_id': framework_id,
            'rating': rating,
            'query': query,
            'time': datetime.now().isoformat()
        }
        st.session_state[key].append(entry)

    def has_given_feedback(self, framework_id: int) -> bool:
        """
        Check if feedback was given for a framework.

        Args:
            framework_id: Framework ID

        Returns:
            True if feedback was given
        """
        feedback = st.session_state[self._get_key('feedback_given')]
        return any(f['framework_id'] == framework_id for f in feedback)

    def get_feedback_for_framework(self, framework_id: int) -> Optional[int]:
        """
        Get the rating given to a framework.

        Args:
            framework_id: Framework ID

        Returns:
            Rating (1 or -1) or None if no feedback
        """
        feedback = st.session_state[self._get_key('feedback_given')]
        for f in feedback:
            if f['framework_id'] == framework_id:
                return f['rating']
        return None

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current session.

        Returns:
            Dict with session statistics
        """
        return {
            'start_time': st.session_state[self._get_key('session_start')],
            'query_count': self.get_query_count(),
            'frameworks_viewed': len(self.get_frameworks_viewed()),
            'messages_count': len(self.get_conversation_history()),
            'current_stage': self.get_current_stage(),
            'feedback_count': len(st.session_state[self._get_key('feedback_given')])
        }

    def clear_session(self) -> None:
        """Clear all session state and reinitialize."""
        keys_to_clear = [
            'conversation_history',
            'frameworks_viewed',
            'last_query',
            'current_stage',
            'selected_framework',
            'selected_framework_id',
            'diagnostic_answers',
            'comparison_frameworks',
            'query_count',
            'feedback_given',
        ]

        for key in keys_to_clear:
            full_key = self._get_key(key)
            if full_key in st.session_state:
                del st.session_state[full_key]

        # Update session start time
        st.session_state[self._get_key('session_start')] = datetime.now().isoformat()

        # Reinitialize
        self._initialize_session()


def init_session_state() -> SessionManager:
    """
    Initialize and return a session manager.

    Convenience function for app startup.

    Returns:
        SessionManager instance
    """
    return SessionManager()
