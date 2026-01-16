"""
Framework Loader Module - Complete Database Version
"""

import os
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import sqlite3


def load_frameworks(db_path: Optional[str] = None) -> pd.DataFrame:
    """Load frameworks from SQLite database"""
    
    if db_path is None:
        db_path = os.getenv('DATABASE_PATH', 'frameworks.db')
    
    db_path = Path(db_path)
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    
    query = """
        SELECT 
            id,
            framework_name as name,
            business_function as business_domains,
            framework_type as type,
            sub_category,
            diagnostic_questions as problem_symptoms,
            levers as use_case,
            diagnostic_questions,
            red_flag_indicators,
            levers,
            current_state_assessment,
            priority_level,
            related_canon,
            skills_required,
            lifecycle_stages,
            notes
        FROM frameworks
        ORDER BY id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        raise ValueError("Database is empty. No frameworks found.")
    
    text_columns = ['name', 'business_domains', 'problem_symptoms', 'use_case', 'levers', 'red_flag_indicators']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    if 'tags' not in df.columns:
        df['tags'] = df['sub_category'].fillna('')
    
    df['searchable_text'] = (
        'Framework: ' + df['name'] + ' | ' +
        'Domain: ' + df['business_domains'] + ' | ' +
        'Problem: ' + df['problem_symptoms'] + ' | ' +
        'Levers: ' + df['levers'] + ' | ' +
        'Red Flags: ' + df['red_flag_indicators']
    )
    
    if 'difficulty_level' not in df.columns:
        df['difficulty_level'] = 'intermediate'
    
    print(f"✓ Loaded {len(df)} frameworks from database: {db_path}")
    return df


def validate_frameworks(df: pd.DataFrame) -> bool:
    required_columns = ['id', 'name', 'business_domains', 'searchable_text']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def get_file_timestamp(file_path: Optional[str] = None) -> float:
    if file_path is None:
        file_path = os.getenv('DATABASE_PATH', 'frameworks.db')
    db_path = Path(file_path)
    if not db_path.exists():
        return 0.0
    return db_path.stat().st_mtime


def get_unique_domains(df: pd.DataFrame) -> List[str]:
    if 'business_domains' in df.columns:
        return sorted(df['business_domains'].unique().tolist())
    return []


def get_unique_types(df: pd.DataFrame) -> List[str]:
    if 'type' in df.columns:
        return sorted(df['type'].unique().tolist())
    return []


def get_unique_subcategories(df: pd.DataFrame) -> List[str]:
    if 'sub_category' in df.columns:
        return sorted(df['sub_category'].dropna().unique().tolist())
    return []


def filter_by_domain(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    if domain and 'business_domains' in df.columns:
        return df[df['business_domains'] == domain]
    return df


def filter_by_type(df: pd.DataFrame, ftype: str) -> pd.DataFrame:
    if ftype and 'type' in df.columns:
        return df[df['type'] == ftype]
    return df


def filter_by_difficulty(df: pd.DataFrame, difficulty: str) -> pd.DataFrame:
    if difficulty and 'difficulty_level' in df.columns:
        return df[df['difficulty_level'] == difficulty]
    return df


def get_framework_by_id(df: pd.DataFrame, framework_id: int) -> Optional[Dict]:
    result = df[df['id'] == framework_id]
    if len(result) > 0:
        return result.iloc[0].to_dict()
    return None


def get_framework_by_name(df: pd.DataFrame, name: str) -> Optional[Dict]:
    result = df[df['name'] == name]
    if len(result) > 0:
        return result.iloc[0].to_dict()
    return None


def search_frameworks(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df
    
    query_lower = query.lower()
    mask = (
        df['name'].str.lower().str.contains(query_lower, na=False) |
        df['business_domains'].str.lower().str.contains(query_lower, na=False) |
        df['problem_symptoms'].str.lower().str.contains(query_lower, na=False) |
        df['searchable_text'].str.lower().str.contains(query_lower, na=False)
    )
    
    return df[mask]


def get_difficulty_levels(df: pd.DataFrame) -> List[str]:
    """Get unique difficulty levels from frameworks"""
    if 'difficulty_level' in df.columns:
        levels = df['difficulty_level'].dropna().unique().tolist()
        # Return in order: beginner, intermediate, advanced
        order = ['beginner', 'intermediate', 'advanced']
        return [l for l in order if l in levels]
    return ['intermediate']
