"""
Framework Loader Module.

This module handles loading and validating the Excel framework database.
It ensures all required fields are present and creates searchable text.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


# Required columns that must be present in the Excel file
REQUIRED_COLUMNS = [
    'id',
    'name',
    'type',
    'business_domains',
    'problem_symptoms',
    'use_case'
]

# Optional columns with default values
OPTIONAL_COLUMNS = {
    'inputs_required': '',
    'outputs_artifacts': '',
    'diagnostic_questions': '',
    'red_flag_indicators': '',
    'levers': '',
    'tags': '',
    'difficulty_level': 'intermediate',
    'related_frameworks': ''
}


def validate_frameworks(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the DataFrame has all required columns.

    Args:
        df: DataFrame loaded from Excel

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check for required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")

    # Check for empty required fields
    if 'id' in df.columns:
        null_ids = df['id'].isna().sum()
        if null_ids > 0:
            errors.append(f"Found {null_ids} rows with missing 'id' values")

    if 'name' in df.columns:
        null_names = df['name'].isna().sum()
        if null_names > 0:
            errors.append(f"Found {null_names} rows with missing 'name' values")

    # Check for duplicate IDs
    if 'id' in df.columns:
        duplicate_ids = df['id'].duplicated().sum()
        if duplicate_ids > 0:
            errors.append(f"Found {duplicate_ids} duplicate 'id' values")

    is_valid = len(errors) == 0
    return is_valid, errors


def create_searchable_text(row: pd.Series) -> str:
    """
    Create searchable text by concatenating relevant fields.

    Args:
        row: A single row from the frameworks DataFrame

    Returns:
        Concatenated searchable text string
    """
    fields = ['name', 'business_domains', 'problem_symptoms', 'use_case', 'tags']
    parts = []

    for field in fields:
        value = row.get(field, '')
        if pd.notna(value) and str(value).strip():
            parts.append(str(value).strip())

    return ' | '.join(parts)


def load_frameworks(
    file_path: Optional[str] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load frameworks from Excel file and prepare for use.

    Args:
        file_path: Path to the Excel file. If None, uses env var or default.
        validate: Whether to validate required columns.

    Returns:
        Cleaned and prepared DataFrame with frameworks.

    Raises:
        FileNotFoundError: If the Excel file doesn't exist.
        ValueError: If validation fails and validate=True.
        pd.errors.EmptyDataError: If the file is empty.
    """
    # Determine file path
    if file_path is None:
        file_path = os.getenv('FRAMEWORKS_FILE', 'data/frameworks.xlsx')

    # Convert to Path object for cross-platform compatibility
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Framework file not found: {file_path}\n"
            f"Please ensure the file exists at the specified location."
        )

    # Load Excel file
    try:
        df = pd.read_excel(
            file_path,
            engine='openpyxl',
            dtype={'id': int}
        )
    except Exception as e:
        # Try with different encoding or engine
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
        except Exception as inner_e:
            raise ValueError(
                f"Failed to load Excel file: {file_path}\n"
                f"Error: {str(inner_e)}\n"
                f"Ensure the file is a valid .xlsx format."
            )

    # Check if DataFrame is empty
    if df.empty:
        raise pd.errors.EmptyDataError(
            f"The framework file is empty: {file_path}"
        )

    # Validate if requested
    if validate:
        is_valid, errors = validate_frameworks(df)
        if not is_valid:
            raise ValueError(
                f"Framework validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

    # Add optional columns with defaults if missing
    for col, default in OPTIONAL_COLUMNS.items():
        if col not in df.columns:
            df[col] = default
        else:
            # Fill NaN values with defaults
            df[col] = df[col].fillna(default)

    # Convert all string columns to proper strings
    string_columns = REQUIRED_COLUMNS[1:] + list(OPTIONAL_COLUMNS.keys())
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', '')

    # Ensure id is integer
    if 'id' in df.columns:
        df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)

    # Create searchable text field
    df['searchable_text'] = df.apply(create_searchable_text, axis=1)

    # Clean up whitespace in all string fields
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    return df


def get_framework_by_id(df: pd.DataFrame, framework_id: int) -> Optional[pd.Series]:
    """
    Get a single framework by its ID.

    Args:
        df: Frameworks DataFrame
        framework_id: The ID to look up

    Returns:
        Series with framework data, or None if not found
    """
    result = df[df['id'] == framework_id]
    if result.empty:
        return None
    return result.iloc[0]


def get_framework_by_name(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """
    Get a single framework by its name (case-insensitive).

    Args:
        df: Frameworks DataFrame
        name: The name to look up

    Returns:
        Series with framework data, or None if not found
    """
    result = df[df['name'].str.lower() == name.lower()]
    if result.empty:
        return None
    return result.iloc[0]


def get_unique_domains(df: pd.DataFrame) -> List[str]:
    """
    Extract all unique business domains from the DataFrame.

    Args:
        df: Frameworks DataFrame

    Returns:
        Sorted list of unique domains
    """
    all_domains = set()

    for domains_str in df['business_domains']:
        if pd.notna(domains_str) and domains_str.strip():
            domains = [d.strip() for d in str(domains_str).split(',')]
            all_domains.update(d for d in domains if d)

    return sorted(all_domains)


def get_unique_types(df: pd.DataFrame) -> List[str]:
    """
    Extract all unique framework types from the DataFrame.

    Args:
        df: Frameworks DataFrame

    Returns:
        Sorted list of unique types
    """
    types = df['type'].dropna().unique()
    return sorted([str(t).strip() for t in types if str(t).strip()])


def get_difficulty_levels(df: pd.DataFrame) -> List[str]:
    """
    Extract all unique difficulty levels from the DataFrame.

    Args:
        df: Frameworks DataFrame

    Returns:
        List of difficulty levels in order
    """
    standard_levels = ['beginner', 'intermediate', 'advanced']
    found_levels = df['difficulty_level'].dropna().unique()

    # Return standard levels that exist in data, plus any custom ones
    result = [l for l in standard_levels if l in found_levels]
    custom = [str(l).strip() for l in found_levels if str(l).strip() not in standard_levels]

    return result + sorted(custom)


def get_file_timestamp(file_path: Optional[str] = None) -> float:
    """
    Get the modification timestamp of the frameworks file.

    Args:
        file_path: Path to check. If None, uses env var or default.

    Returns:
        Modification timestamp as float
    """
    if file_path is None:
        file_path = os.getenv('FRAMEWORKS_FILE', 'data/frameworks.xlsx')

    path = Path(file_path)
    if not path.exists():
        return 0.0

    return path.stat().st_mtime
