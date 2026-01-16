"""
Semantic Search Module.

This module provides semantic search functionality over framework embeddings
using cosine similarity for ranking and optional filtering.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


class SearchResult:
    """
    Represents a single search result.

    Attributes:
        framework_id: ID of the matching framework
        score: Similarity score (0-1)
        framework_data: Full framework data as dict
    """

    def __init__(
        self,
        framework_id: int,
        score: float,
        framework_data: Dict[str, Any]
    ):
        self.framework_id = framework_id
        self.score = score
        self.framework_data = framework_data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'framework_id': self.framework_id,
            'score': self.score,
            **self.framework_data
        }

    def __repr__(self) -> str:
        name = self.framework_data.get('name', 'Unknown')
        return f"SearchResult(id={self.framework_id}, score={self.score:.3f}, name='{name}')"


class SemanticSearch:
    """
    Performs semantic search over framework embeddings.

    Attributes:
        embeddings: Dict mapping framework IDs to embeddings
        frameworks_df: DataFrame with framework data
        top_k: Default number of results to return
        min_similarity: Minimum similarity threshold
    """

    def __init__(
        self,
        embeddings: Dict[int, np.ndarray],
        frameworks_df: pd.DataFrame,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None
    ):
        """
        Initialize the semantic search engine.

        Args:
            embeddings: Dict mapping framework IDs to embedding arrays
            frameworks_df: DataFrame with framework data
            top_k: Default number of results (from env or 5)
            min_similarity: Minimum similarity threshold (from env or 0.6)
        """
        self.embeddings = embeddings
        self.frameworks_df = frameworks_df
        self.top_k = top_k or int(os.getenv('SEARCH_TOP_K', 5))
        self.min_similarity = min_similarity or float(os.getenv('SEARCH_MIN_SIMILARITY', 0.6))

        # Pre-compute embedding matrix for efficient search
        self._build_index()

    def _build_index(self) -> None:
        """Build embedding matrix and ID mapping for efficient search."""
        valid_ids = [
            int(id) for id in self.frameworks_df['id'].tolist()
            if id in self.embeddings
        ]

        if valid_ids:
            self._embedding_matrix = np.vstack([
                self.embeddings[id] for id in valid_ids
            ])
            self._id_mapping = valid_ids
        else:
            self._embedding_matrix = np.array([])
            self._id_mapping = []

    def _cosine_similarity(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and corpus.

        Args:
            query_embedding: Query embedding vector
            corpus_embeddings: Matrix of corpus embeddings

        Returns:
            Array of similarity scores
        """
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Normalize corpus (row-wise)
        corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        corpus_normalized = corpus_embeddings / corpus_norms

        # Compute dot product (equals cosine similarity for normalized vectors)
        similarities = np.dot(corpus_normalized, query_norm)

        return similarities

    def _filter_by_domain(
        self,
        ids: List[int],
        domains: List[str]
    ) -> List[int]:
        """
        Filter framework IDs by domain (AND logic).

        Args:
            ids: List of framework IDs
            domains: List of domains to filter by

        Returns:
            Filtered list of IDs
        """
        if not domains:
            return ids

        filtered_ids = []
        for framework_id in ids:
            row = self.frameworks_df[self.frameworks_df['id'] == framework_id]
            if row.empty:
                continue

            framework_domains = str(row.iloc[0]['business_domains']).lower()

            # AND logic: all specified domains must be present
            matches_all = all(
                domain.lower() in framework_domains
                for domain in domains
            )

            if matches_all:
                filtered_ids.append(framework_id)

        return filtered_ids

    def _filter_by_difficulty(
        self,
        ids: List[int],
        difficulty: str
    ) -> List[int]:
        """
        Filter framework IDs by difficulty level.

        Args:
            ids: List of framework IDs
            difficulty: Difficulty level to filter by

        Returns:
            Filtered list of IDs
        """
        if not difficulty:
            return ids

        filtered_ids = []
        for framework_id in ids:
            row = self.frameworks_df[self.frameworks_df['id'] == framework_id]
            if row.empty:
                continue

            framework_difficulty = str(row.iloc[0]['difficulty_level']).lower()

            if framework_difficulty == difficulty.lower():
                filtered_ids.append(framework_id)

        return filtered_ids

    def _filter_by_type(
        self,
        ids: List[int],
        framework_type: str
    ) -> List[int]:
        """
        Filter framework IDs by type.

        Args:
            ids: List of framework IDs
            framework_type: Type to filter by

        Returns:
            Filtered list of IDs
        """
        if not framework_type:
            return ids

        filtered_ids = []
        for framework_id in ids:
            row = self.frameworks_df[self.frameworks_df['id'] == framework_id]
            if row.empty:
                continue

            fw_type = str(row.iloc[0]['type']).lower()

            if fw_type == framework_type.lower():
                filtered_ids.append(framework_id)

        return filtered_ids

    def _get_framework_data(self, framework_id: int) -> Dict[str, Any]:
        """
        Get framework data as dictionary.

        Args:
            framework_id: Framework ID

        Returns:
            Framework data as dict
        """
        row = self.frameworks_df[self.frameworks_df['id'] == framework_id]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        domains: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
        framework_type: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for frameworks matching the query.

        Args:
            query_embedding: Embedding of the query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            domains: List of domains to filter by (AND logic)
            difficulty: Difficulty level to filter by
            framework_type: Framework type to filter by

        Returns:
            List of SearchResult objects, sorted by score descending
        """
        top_k = top_k or self.top_k
        min_similarity = min_similarity if min_similarity is not None else self.min_similarity

        if len(self._embedding_matrix) == 0:
            return []

        # Compute similarities
        similarities = self._cosine_similarity(query_embedding, self._embedding_matrix)

        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]

        # Build results list
        results = []
        for idx in sorted_indices:
            score = float(similarities[idx])
            framework_id = self._id_mapping[idx]

            # Skip if below threshold
            if score < min_similarity:
                continue

            results.append((framework_id, score))

        # Apply filters
        filtered_ids = [r[0] for r in results]

        if domains:
            filtered_ids = self._filter_by_domain(filtered_ids, domains)

        if difficulty:
            filtered_ids = self._filter_by_difficulty(filtered_ids, difficulty)

        if framework_type:
            filtered_ids = self._filter_by_type(filtered_ids, framework_type)

        # Convert to SearchResult objects
        id_to_score = {r[0]: r[1] for r in results}
        search_results = []

        for framework_id in filtered_ids:
            if framework_id in id_to_score:
                score = id_to_score[framework_id]
                framework_data = self._get_framework_data(framework_id)
                search_results.append(SearchResult(framework_id, score, framework_data))

        # Sort by score and limit to top_k
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:top_k]

    def search_by_text(
        self,
        query: str,
        embed_func,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search using a text query (convenience method).

        Args:
            query: Text query to search for
            embed_func: Function to embed the query text
            **kwargs: Additional arguments passed to search()

        Returns:
            List of SearchResult objects
        """
        query_embedding = embed_func(query)
        return self.search(query_embedding, **kwargs)

    def get_related_frameworks(
        self,
        framework_id: int,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Get frameworks related to a given framework.

        Args:
            framework_id: Framework to find related frameworks for
            top_k: Number of results to return

        Returns:
            List of SearchResult objects (excluding the input framework)
        """
        top_k = top_k or self.top_k

        # Get embedding for the framework
        if framework_id not in self.embeddings:
            return []

        query_embedding = self.embeddings[framework_id]

        # Search without minimum threshold to get more results
        results = self.search(query_embedding, top_k=top_k + 1, min_similarity=0.0)

        # Remove the queried framework from results
        return [r for r in results if r.framework_id != framework_id][:top_k]

    def get_framework_by_name(self, name: str) -> Optional[SearchResult]:
        """
        Get a framework by exact name match (case-insensitive).

        Args:
            name: Framework name to search for

        Returns:
            SearchResult if found, None otherwise
        """
        row = self.frameworks_df[
            self.frameworks_df['name'].str.lower() == name.lower()
        ]

        if row.empty:
            return None

        framework_id = int(row.iloc[0]['id'])
        return SearchResult(
            framework_id=framework_id,
            score=1.0,
            framework_data=row.iloc[0].to_dict()
        )

    def get_framework_by_id(self, framework_id: int) -> Optional[SearchResult]:
        """
        Get a framework by ID.

        Args:
            framework_id: Framework ID

        Returns:
            SearchResult if found, None otherwise
        """
        row = self.frameworks_df[self.frameworks_df['id'] == framework_id]

        if row.empty:
            return None

        return SearchResult(
            framework_id=framework_id,
            score=1.0,
            framework_data=row.iloc[0].to_dict()
        )

    def get_frameworks_by_ids(self, framework_ids: List[int]) -> List[SearchResult]:
        """
        Get multiple frameworks by their IDs.

        Args:
            framework_ids: List of framework IDs

        Returns:
            List of SearchResult objects
        """
        results = []
        for framework_id in framework_ids:
            result = self.get_framework_by_id(framework_id)
            if result:
                results.append(result)
        return results

    def fuzzy_name_search(
        self,
        query: str,
        threshold: float = 0.6
    ) -> List[SearchResult]:
        """
        Search for frameworks by name similarity.

        Args:
            query: Name query
            threshold: Minimum similarity threshold

        Returns:
            List of SearchResult objects
        """
        query_lower = query.lower()
        results = []

        for _, row in self.frameworks_df.iterrows():
            name = str(row['name']).lower()

            # Simple substring match
            if query_lower in name or name in query_lower:
                score = len(query_lower) / max(len(name), len(query_lower))
                results.append(SearchResult(
                    framework_id=int(row['id']),
                    score=score,
                    framework_data=row.to_dict()
                ))

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results


def search_frameworks(df, query: str, top_k: int = 5):
    """
    Standalone search function wrapper for compatibility.
    
    Args:
        df: DataFrame with frameworks
        query: Search query string
        top_k: Number of results to return
    
    Returns:
        DataFrame with top matching frameworks
    """
    # Simple text search as fallback
    if not query:
        return df.head(top_k)
    
    query_lower = query.lower()
    mask = (
        df['name'].str.lower().str.contains(query_lower, case=False, na=False) |
        df['business_domains'].str.lower().str.contains(query_lower, case=False, na=False) |
        df['searchable_text'].str.lower().str.contains(query_lower, case=False, na=False)
    )
    
    results = df[mask]
    return results.head(top_k) if len(results) > 0 else df.head(0)
