#!/usr/bin/env python3
"""
Setup Validation Script for Framework Assistant.

This script validates that all requirements are met to run
the Framework Assistant application.
"""

import os
import sys
from pathlib import Path


# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 50}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 50}{Colors.END}")


def print_check(name: str, passed: bool, message: str = "") -> None:
    """Print a check result."""
    if passed:
        status = f"{Colors.GREEN}PASS{Colors.END}"
    else:
        status = f"{Colors.RED}FAIL{Colors.END}"

    print(f"  [{status}] {name}")
    if message and not passed:
        print(f"         {Colors.YELLOW}{message}{Colors.END}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"  {Colors.YELLOW}WARNING: {message}{Colors.END}")


def check_python_version() -> bool:
    """Check Python version is 3.9+."""
    version = sys.version_info
    passed = version.major >= 3 and version.minor >= 9

    version_str = f"{version.major}.{version.minor}.{version.micro}"
    message = f"Python 3.9+ required, found {version_str}" if not passed else ""

    print_check(f"Python version ({version_str})", passed, message)
    return passed


def check_directory_structure() -> bool:
    """Check that required directories exist."""
    required_dirs = [
        'utils',
        'handlers',
        'data',
        'cache',
        'logs'
    ]

    all_exist = True
    for dir_name in required_dirs:
        exists = Path(dir_name).is_dir()
        print_check(f"Directory '{dir_name}/'", exists)
        if not exists:
            all_exist = False

    return all_exist


def check_required_files() -> bool:
    """Check that required files exist."""
    required_files = [
        ('app.py', True),
        ('requirements.txt', True),
        ('.env', True),
        ('utils/__init__.py', True),
        ('utils/loader.py', True),
        ('utils/embedder.py', True),
        ('utils/search.py', True),
        ('utils/llm.py', True),
        ('utils/intent.py', True),
        ('utils/session.py', True),
        ('handlers/__init__.py', True),
        ('handlers/diagnostic.py', True),
        ('handlers/discovery.py', True),
        ('handlers/details.py', True),
        ('handlers/sequencing.py', True),
        ('handlers/comparison.py', True),
        ('data/frameworks.xlsx', False),  # Optional - user provides
    ]

    all_required_exist = True

    for file_path, required in required_files:
        exists = Path(file_path).is_file()

        if required:
            print_check(f"File '{file_path}'", exists)
            if not exists:
                all_required_exist = False
        else:
            if exists:
                print_check(f"File '{file_path}' (optional)", True)
            else:
                print_warning(f"Optional file '{file_path}' not found")

    return all_required_exist


def check_dependencies() -> bool:
    """Check that required Python packages are installed."""
    required_packages = [
        ('streamlit', 'streamlit'),
        ('openai', 'openai'),
        ('pandas', 'pandas'),
        ('openpyxl', 'openpyxl'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('dotenv', 'python-dotenv'),
        ('tqdm', 'tqdm'),
    ]

    all_installed = True

    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print_check(f"Package '{package_name}'", True)
        except ImportError:
            print_check(f"Package '{package_name}'", False, f"pip install {package_name}")
            all_installed = False

    return all_installed


def check_environment_variables() -> bool:
    """Check that required environment variables are set."""
    from dotenv import load_dotenv
    load_dotenv()

    required_vars = [
        ('OPENAI_API_KEY', True),
    ]

    optional_vars = [
        'OPENAI_EMBEDDING_MODEL',
        'OPENAI_LLM_MODEL',
        'FRAMEWORKS_FILE',
    ]

    all_required_set = True

    for var_name, required in required_vars:
        value = os.getenv(var_name)
        is_set = value is not None and len(value) > 0

        if required:
            # Mask API key for display
            display_value = value[:8] + "..." if is_set and 'KEY' in var_name else ""
            print_check(f"Env var '{var_name}' {display_value}", is_set)
            if not is_set:
                all_required_set = False
        else:
            if is_set:
                print_check(f"Env var '{var_name}' (optional)", True)

    for var_name in optional_vars:
        value = os.getenv(var_name)
        if value:
            print_check(f"Env var '{var_name}' (optional)", True)

    return all_required_set


def check_openai_connection() -> bool:
    """Test OpenAI API connection."""
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print_check("OpenAI API connection", False, "API key not set")
            return False

        client = OpenAI(api_key=api_key)

        # Test with a minimal API call
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test",
            dimensions=256
        )

        print_check("OpenAI API connection", True)
        return True

    except Exception as e:
        print_check("OpenAI API connection", False, str(e)[:50])
        return False


def check_framework_file() -> bool:
    """Check if framework file is loadable."""
    from dotenv import load_dotenv
    load_dotenv()

    frameworks_file = os.getenv('FRAMEWORKS_FILE', 'data/frameworks.xlsx')

    if not Path(frameworks_file).exists():
        print_warning(f"Framework file '{frameworks_file}' not found")
        print_warning("You'll need to add your framework data before running the app")
        return True  # Not a failure, just a warning

    try:
        import pandas as pd
        df = pd.read_excel(frameworks_file, engine='openpyxl')

        # Check required columns
        required_cols = ['id', 'name', 'type', 'business_domains', 'problem_symptoms', 'use_case']
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            print_check(
                f"Framework file structure",
                False,
                f"Missing columns: {', '.join(missing)}"
            )
            return False

        print_check(f"Framework file ({len(df)} frameworks)", True)
        return True

    except Exception as e:
        print_check("Framework file loadable", False, str(e)[:50])
        return False


def main() -> int:
    """Run all checks and return exit code."""
    print(f"\n{Colors.BOLD}Framework Assistant Setup Checker{Colors.END}")
    print("=" * 40)

    all_passed = True

    # Python version
    print_header("Python Version")
    if not check_python_version():
        all_passed = False

    # Directory structure
    print_header("Directory Structure")
    if not check_directory_structure():
        all_passed = False

    # Required files
    print_header("Required Files")
    if not check_required_files():
        all_passed = False

    # Dependencies
    print_header("Dependencies")
    if not check_dependencies():
        all_passed = False
        print(f"\n  {Colors.YELLOW}Install missing packages with:{Colors.END}")
        print(f"  pip install -r requirements.txt")

    # Environment variables
    print_header("Environment Variables")
    if not check_environment_variables():
        all_passed = False
        print(f"\n  {Colors.YELLOW}Configure your .env file:{Colors.END}")
        print(f"  cp .env.example .env")
        print(f"  # Edit .env and add your OpenAI API key")

    # OpenAI connection
    print_header("OpenAI API Connection")
    if not check_openai_connection():
        all_passed = False

    # Framework file
    print_header("Framework Database")
    check_framework_file()

    # Summary
    print_header("Summary")
    if all_passed:
        print(f"  {Colors.GREEN}{Colors.BOLD}All checks passed!{Colors.END}")
        print(f"\n  Run the application with:")
        print(f"  {Colors.BOLD}streamlit run app.py{Colors.END}")
        return 0
    else:
        print(f"  {Colors.RED}{Colors.BOLD}Some checks failed.{Colors.END}")
        print(f"  Please fix the issues above and run this script again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
