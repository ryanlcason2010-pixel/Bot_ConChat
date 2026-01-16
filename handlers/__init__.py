"""
Handlers package for Framework Assistant.

This package contains pattern handlers for:
- diagnostic: Symptom → Diagnostic flow
- discovery: Framework discovery
- details: Framework details
- sequencing: Framework sequencing
- comparison: Framework comparison
"""

from .diagnostic import handle_diagnostic
from .discovery import handle_discovery
from .details import handle_details
from .sequencing import handle_sequencing
from .comparison import handle_comparison

__all__ = [
    'handle_diagnostic',
    'handle_discovery',
    'handle_details',
    'handle_sequencing',
    'handle_comparison',
]
