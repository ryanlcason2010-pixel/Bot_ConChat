"""
Framework Assistant - Main Streamlit Application.

An adaptive AI tool that helps consultants navigate their proprietary
framework library through natural conversation.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import streamlit as st
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.loader import (
    load_frameworks,
    get_unique_domains,
    get_unique_types,
    get_difficulty_levels
)
from utils.embedder import EmbeddingEngine
from utils.search import SemanticSearch
from utils.llm import LLMClient
from utils.intent import (
    detect_intent,
    extract_framework_names,
    INTENT_DIAGNOSTIC,
    INTENT_DISCOVERY,
    INTENT_DETAILS,
    INTENT_SEQUENCING,
    INTENT_COMPARISON,
    INTENT_UNKNOWN
)
from utils.session import SessionManager, init_session_state

from handlers.diagnostic import (
    handle_diagnostic,
    handle_framework_selection,
    handle_diagnostic_analysis,
    format_diagnostic_card
)
from handlers.discovery import handle_discovery, format_discovery_card
from handlers.details import handle_details, handle_details_by_name, format_framework_card_detailed
from handlers.sequencing import handle_sequencing, handle_sequencing_by_name
from handlers.comparison import handle_comparison, handle_comparison_by_names


# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Framework Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .user-message {
        background-color: #e3f2fd;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        text-align: right;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
    }
    .framework-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: white;
    }
    .difficulty-beginner { color: #4caf50; }
    .difficulty-intermediate { color: #ff9800; }
    .difficulty-advanced { color: #f44336; }
    .feedback-btn {
        cursor: pointer;
        font-size: 1.2em;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def log_feedback(query: str, framework_id: int, rating: int) -> None:
    """
    Log user feedback to JSON file.

    Args:
        query: The query that led to this framework
        framework_id: ID of the framework
        rating: 1 for thumbs up, -1 for thumbs down
    """
    feedback_file = os.getenv('FEEDBACK_LOG_FILE', 'logs/feedback.json')

    feedback = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "framework_id": framework_id,
        "rating": rating
    }

    try:
        # Ensure directory exists
        Path(feedback_file).parent.mkdir(parents=True, exist_ok=True)

        # Read existing feedback
        if Path(feedback_file).exists():
            with open(feedback_file, 'r') as f:
                content = f.read().strip()
                if content:
                    existing = json.loads(content)
                else:
                    existing = []
        else:
            existing = []

        # Append new feedback
        existing.append(feedback)

        # Write back
        with open(feedback_file, 'w') as f:
            json.dump(existing, f, indent=2)

    except Exception as e:
        st.warning(f"Failed to log feedback: {e}")


@st.cache_resource
def load_resources():
    """
    Load and cache all resources.

    Returns:
        Tuple of (frameworks_df, embedding_engine, llm_client)
    """
    frameworks_file = os.getenv('DATABASE_PATH', 'frameworks.db')

    try:
        # Load frameworks
        df = load_frameworks(frameworks_file)

        # Initialize embedding engine
        embedding_engine = EmbeddingEngine()

        # Initialize LLM client
        llm_client = LLMClient()

        return df, embedding_engine, llm_client

    except FileNotFoundError:
        st.error(f"""
        **Framework file not found!**

        Please ensure `{frameworks_file}` exists with your framework data.

        See `data/README.md` for the required schema.
        """)
        st.stop()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()


def init_app_state():
    """Initialize all application state."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.show_frameworks = []
        st.session_state.pending_selection = False
        st.session_state.available_frameworks = []


def render_sidebar(
    df,
    session: SessionManager,
    domains: List[str],
    types: List[str],
    difficulties: List[str]
):
    """Render the sidebar with filters and session info."""
    with st.sidebar:
        st.title("Framework Assistant")
        st.markdown("---")

        # Stats
        st.metric("Frameworks Loaded", len(df))

        # Filters
        st.subheader("Filters")

        selected_domains = st.multiselect(
            "Business Domains",
            options=domains,
            default=[],
            key="domain_filter"
        )

        selected_difficulty = st.radio(
            "Difficulty Level",
            options=["All"] + difficulties,
            key="difficulty_filter"
        )

        selected_type = st.selectbox(
            "Framework Type",
            options=["All"] + types,
            key="type_filter"
        )

        st.markdown("---")

        # Session info
        st.subheader("Session Info")
        summary = session.get_session_summary()

        st.text(f"Queries: {summary['query_count']}")
        st.text(f"Frameworks viewed: {summary['frameworks_viewed']}")
        st.text(f"Messages: {summary['messages_count']}")

        if st.button("Start Fresh", type="secondary"):
            session.clear_session()
            st.session_state.show_frameworks = []
            st.session_state.pending_selection = False
            st.rerun()

        st.markdown("---")
        st.caption("Powered by OpenAI GPT-4o-mini")

    return selected_domains, selected_difficulty, selected_type


def render_chat_history(session: SessionManager):
    """Render the conversation history."""
    history = session.get_conversation_history()

    for msg in history:
        role = msg['role']
        content = msg['content']

        if role == 'user':
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)


def render_framework_card(
    framework: Dict[str, Any],
    session: SessionManager,
    expanded: bool = False
):
    """
    Render a framework card with details and feedback buttons.

    Args:
        framework: Framework data
        session: Session manager
        expanded: Whether to show expanded view
    """
    framework_id = framework.get('id', 0)
    name = framework.get('name', 'Unknown')
    difficulty = framework.get('difficulty_level', 'intermediate')
    domains = framework.get('business_domains', '')
    use_case = framework.get('use_case', '')

    # Difficulty color
    diff_color = {
        'beginner': 'green',
        'intermediate': 'orange',
        'advanced': 'red'
    }.get(difficulty.lower(), 'gray')

    with st.expander(f"**{name}** - :{diff_color}[{difficulty}]", expanded=expanded):
        st.markdown(f"**Domains:** {domains}")
        st.markdown(f"**Use Case:** {use_case}")

        if expanded:
            # Show additional details
            if framework.get('diagnostic_questions'):
                st.markdown("**Diagnostic Questions:**")
                questions = framework['diagnostic_questions'].split('|')
                for q in questions:
                    if q.strip():
                        st.markdown(f"- {q.strip()}")

            if framework.get('red_flag_indicators'):
                st.markdown(f"**Red Flags:** {framework['red_flag_indicators']}")

            if framework.get('levers'):
                st.markdown(f"**Levers:** {framework['levers']}")

        # Feedback buttons
        col1, col2, col3 = st.columns([1, 1, 4])

        # Check if already gave feedback
        existing_rating = session.get_feedback_for_framework(framework_id)

        with col1:
            if st.button("👍", key=f"up_{framework_id}",
                        disabled=existing_rating is not None):
                log_feedback(session.get_last_query(), framework_id, 1)
                session.add_feedback(framework_id, 1, session.get_last_query())
                st.success("Thanks!")
                st.rerun()

        with col2:
            if st.button("👎", key=f"down_{framework_id}",
                        disabled=existing_rating is not None):
                log_feedback(session.get_last_query(), framework_id, -1)
                session.add_feedback(framework_id, -1, session.get_last_query())
                st.info("Thanks for feedback")
                st.rerun()

        if existing_rating:
            with col3:
                emoji = "👍" if existing_rating > 0 else "👎"
                st.caption(f"You rated: {emoji}")


def process_query(
    query: str,
    session: SessionManager,
    search_engine: SemanticSearch,
    llm_client: LLMClient,
    embedding_engine: EmbeddingEngine,
    df,
    domains_filter: List[str],
    difficulty_filter: Optional[str],
    type_filter: Optional[str]
) -> tuple[str, List[Dict]]:
    """
    Process a user query and generate response.

    Args:
        query: User's query
        session: Session manager
        search_engine: Semantic search engine
        llm_client: LLM client
        embedding_engine: Embedding engine
        df: Frameworks DataFrame
        domains_filter: Domain filters
        difficulty_filter: Difficulty filter
        type_filter: Type filter

    Returns:
        Tuple of (response, frameworks to display)
    """
    # Check for framework selection in diagnostic flow
    current_stage = session.get_current_stage()
    available_frameworks = st.session_state.get('available_frameworks', [])

    if current_stage == 'framework_selection' and available_frameworks:
        # Try to interpret as selection
        selected, msg = handle_framework_selection(query, available_frameworks, session)
        if selected:
            return msg, [selected]

    if current_stage == 'diagnostic_active':
        # Treat input as diagnostic answers
        framework = session.get_selected_framework()
        if framework:
            response = handle_diagnostic_analysis(query, framework, session, llm_client)
            return response, []

    # Detect intent
    known_names = df['name'].tolist()
    intent, confidence = detect_intent(query, known_names)

    # Get query embedding
    query_embedding = embedding_engine.embed_text(query)

    # Apply filters
    difficulty = difficulty_filter if difficulty_filter != "All" else None
    fw_type = type_filter if type_filter != "All" else None

    # Search for frameworks
    search_results = search_engine.search(
        query_embedding,
        domains=domains_filter if domains_filter else None,
        difficulty=difficulty,
        framework_type=fw_type
    )

    # Route to appropriate handler
    if intent == INTENT_DIAGNOSTIC:
        response, frameworks = handle_diagnostic(query, search_results, session, llm_client)
        st.session_state.available_frameworks = frameworks
        return response, frameworks

    elif intent == INTENT_DISCOVERY:
        response, frameworks = handle_discovery(
            query, search_results, llm_client,
            domains_filter, difficulty, len(df)
        )
        return response, frameworks

    elif intent == INTENT_DETAILS:
        # Extract framework name from query
        mentioned = extract_framework_names(query, known_names)
        if mentioned:
            response = handle_details_by_name(mentioned[0], search_engine, llm_client)
            result = search_engine.get_framework_by_name(mentioned[0])
            return response, [result.framework_data] if result else []
        elif search_results:
            # Use top search result
            response = handle_details(search_results[0].framework_id, search_engine, llm_client)
            return response, [search_results[0].framework_data]
        else:
            return "I couldn't identify which framework you're asking about. Could you specify the name?", []

    elif intent == INTENT_SEQUENCING:
        mentioned = extract_framework_names(query, known_names)
        if mentioned:
            response = handle_sequencing_by_name(mentioned[0], search_engine, llm_client)
            result = search_engine.get_framework_by_name(mentioned[0])
            return response, [result.framework_data] if result else []
        elif search_results:
            response = handle_sequencing(search_results[0].framework_id, search_engine, llm_client)
            return response, [search_results[0].framework_data]
        else:
            return "Please specify which framework you'd like to see the sequence for.", []

    elif intent == INTENT_COMPARISON:
        mentioned = extract_framework_names(query, known_names)
        if len(mentioned) >= 2:
            response = handle_comparison_by_names(mentioned[0], mentioned[1], search_engine, llm_client)
            frameworks = []
            for name in mentioned[:2]:
                result = search_engine.get_framework_by_name(name)
                if result:
                    frameworks.append(result.framework_data)
            return response, frameworks
        elif len(mentioned) == 1 and search_results:
            # Compare mentioned framework with top search result
            response = handle_comparison_by_names(
                mentioned[0],
                search_results[0].framework_data.get('name', ''),
                search_engine,
                llm_client
            )
            return response, [search_results[0].framework_data]
        else:
            return "Please specify two frameworks to compare (e.g., 'Compare SPIN Selling vs MEDDIC').", []

    else:
        # Unknown intent - treat as diagnostic
        response, frameworks = handle_diagnostic(query, search_results, session, llm_client)
        st.session_state.available_frameworks = frameworks
        return response, frameworks


def main():
    """Main application entry point."""
    # Initialize state
    init_app_state()
    session = init_session_state()

    # Load resources
    with st.spinner("Loading frameworks..."):
        df, embedding_engine, llm_client = load_resources()

    # Generate embeddings if needed
    frameworks_file = os.getenv('DATABASE_PATH', 'frameworks.db')

    if 'embeddings_loaded' not in st.session_state:
        with st.spinner("Preparing semantic search (this may take a moment on first run)..."):
            embeddings = embedding_engine.get_or_create_embeddings(df, frameworks_file)
            st.session_state.embeddings_loaded = True
            st.session_state.embeddings = embeddings
    else:
        embeddings = st.session_state.embeddings
        embedding_engine.embeddings = embeddings

    # Create search engine
    search_engine = SemanticSearch(embeddings, df)

    # Get filter options
    domains = get_unique_domains(df)
    types = get_unique_types(df)
    difficulties = get_difficulty_levels(df)

    # Render sidebar
    selected_domains, selected_difficulty, selected_type = render_sidebar(
        df, session, domains, types, difficulties
    )

    # Main content area
    st.title("Framework Assistant")
    st.markdown("*Navigate your framework library through natural conversation*")

    # Chat interface
    st.markdown("---")

    # Render chat history
    render_chat_history(session)

    # Show frameworks from last response
    if st.session_state.show_frameworks:
        st.subheader("Relevant Frameworks")
        for fw in st.session_state.show_frameworks:
            render_framework_card(fw, session, expanded=False)

    # Chat input
    user_query = st.chat_input("Describe your client situation or ask about frameworks...")

    if user_query:
        # Add user message
        session.add_message("user", user_query)
        session.set_last_query(user_query)

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Process query
        with st.spinner("Thinking..."):
            response, frameworks = process_query(
                user_query,
                session,
                search_engine,
                llm_client,
                embedding_engine,
                df,
                selected_domains,
                selected_difficulty,
                selected_type
            )

        # Add assistant response
        session.add_message("assistant", response, {
            'frameworks_shown': [f.get('id') for f in frameworks],
            'intent': detect_intent(user_query)[0]
        })

        # Update state
        st.session_state.show_frameworks = frameworks

        # Record viewed frameworks
        for fw in frameworks:
            session.add_framework_viewed(fw.get('name', 'Unknown'), fw.get('id', 0))

        # Display response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Display framework cards
        if frameworks:
            st.subheader("Relevant Frameworks")
            for fw in frameworks:
                render_framework_card(fw, session, expanded=False)

        st.rerun()

    # Help section
    with st.expander("How to use Framework Assistant"):
        st.markdown("""
        **Query Types:**

        1. **Diagnostic** - Describe a problem and get framework recommendations
           - *"Client has high employee turnover"*
           - *"Struggling with low conversion rates"*

        2. **Discovery** - Browse and search frameworks
           - *"Show me all sales frameworks"*
           - *"What frameworks do you have for strategy?"*

        3. **Details** - Learn about a specific framework
           - *"Tell me about SPIN Selling"*
           - *"Explain the MEDDIC framework"*

        4. **Sequencing** - Understand framework order
           - *"What should I do before MEDDIC?"*
           - *"What comes after customer discovery?"*

        5. **Comparison** - Compare two frameworks
           - *"Compare SPIN Selling vs Consultative Selling"*
           - *"What's the difference between OKRs and KPIs?"*

        **Tips:**
        - Use the sidebar filters to narrow down results
        - Click thumbs up/down on frameworks to improve recommendations
        - Click "Start Fresh" to begin a new conversation
        """)


if __name__ == "__main__":
    main()
