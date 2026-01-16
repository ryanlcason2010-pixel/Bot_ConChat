"""
Embedding Engine Module.

This module handles generating and caching embeddings for framework search.
Uses OpenAI's text-embedding-3-small model with intelligent caching.
"""

import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from openai import OpenAI

from .loader import get_file_timestamp


class EmbeddingEngine:
    """
    Manages embedding generation and caching for frameworks.

    Attributes:
        client: OpenAI client instance
        model: Embedding model name
        dimensions: Embedding dimensions
        cache_file: Path to cache file
        embeddings: Dict mapping framework IDs to embeddings
        token_usage: Total tokens used for embeddings
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        dimensions: int = 1536,
        cache_file: Optional[str] = None
    ):
        """
        Initialize the embedding engine.

        Args:
            client: OpenAI client. If None, creates from env var.
            model: Embedding model name. If None, uses env var.
            dimensions: Embedding dimensions.
            cache_file: Path to cache file. If None, uses env var.
        """
        self.client = client or OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model or os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        self.dimensions = int(os.getenv('EMBEDDING_DIMENSIONS', dimensions))
        self.cache_file = Path(
            cache_file or os.getenv('EMBEDDINGS_CACHE_FILE', 'cache/embeddings_cache.pkl')
        )

        self.embeddings: Dict[int, np.ndarray] = {}
        self.token_usage: int = 0
        self._cache_metadata: Dict[str, Any] = {}

    def _embed_text_with_retry(
        self,
        text: str,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> np.ndarray:
        """
        Embed text with exponential backoff retry logic.

        Args:
            text: Text to embed
            max_retries: Maximum retry attempts
            base_delay: Base delay in seconds

        Returns:
            Embedding as numpy array

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    dimensions=self.dimensions
                )

                self.token_usage += response.usage.total_tokens
                return np.array(response.data[0].embedding)

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)

        raise last_exception

    def _embed_batch_with_retry(
        self,
        texts: List[str],
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> List[np.ndarray]:
        """
        Embed multiple texts in a batch with retry logic.

        Args:
            texts: List of texts to embed
            max_retries: Maximum retry attempts
            base_delay: Base delay in seconds

        Returns:
            List of embeddings as numpy arrays
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    dimensions=self.dimensions
                )

                self.token_usage += response.usage.total_tokens
                return [np.array(item.embedding) for item in response.data]

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)

        raise last_exception

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding as numpy array
        """
        return self._embed_text_with_retry(text)

    def embed_frameworks(
        self,
        df: pd.DataFrame,
        batch_size: int = 100,
        show_progress: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        Embed all frameworks in the DataFrame.

        Args:
            df: DataFrame with frameworks (must have 'id' and 'searchable_text')
            batch_size: Number of texts to embed per API call
            show_progress: Whether to show progress bar

        Returns:
            Dict mapping framework IDs to embeddings
        """
        embeddings = {}
        texts = df['searchable_text'].tolist()
        ids = df['id'].tolist()

        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Embedding frameworks")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            try:
                batch_embeddings = self._embed_batch_with_retry(batch_texts)

                for framework_id, embedding in zip(batch_ids, batch_embeddings):
                    embeddings[framework_id] = embedding

            except Exception as e:
                # Fall back to individual embedding if batch fails
                for framework_id, text in zip(batch_ids, batch_texts):
                    try:
                        embedding = self._embed_text_with_retry(text)
                        embeddings[framework_id] = embedding
                    except Exception as inner_e:
                        print(f"Failed to embed framework {framework_id}: {inner_e}")

        self.embeddings = embeddings
        return embeddings

    def _is_cache_valid(self, frameworks_file: Optional[str] = None) -> bool:
        """
        Check if the cache is valid based on file timestamps.

        Args:
            frameworks_file: Path to frameworks file

        Returns:
            True if cache is valid, False otherwise
        """
        if not self.cache_file.exists():
            return False

        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            if not isinstance(cache_data, dict):
                return False

            # Check required keys
            if 'embeddings' not in cache_data or 'metadata' not in cache_data:
                return False

            metadata = cache_data['metadata']

            # Check framework file timestamp
            current_timestamp = get_file_timestamp(frameworks_file)
            cached_timestamp = metadata.get('frameworks_timestamp', 0)

            if current_timestamp != cached_timestamp:
                return False

            # Check model
            if metadata.get('model') != self.model:
                return False

            # Check dimensions
            if metadata.get('dimensions') != self.dimensions:
                return False

            return True

        except Exception:
            return False

    def load_cache(self, frameworks_file: Optional[str] = None) -> bool:
        """
        Load embeddings from cache if valid.

        Args:
            frameworks_file: Path to frameworks file for validation

        Returns:
            True if cache was loaded, False otherwise
        """
        if not self._is_cache_valid(frameworks_file):
            return False

        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            self.embeddings = cache_data['embeddings']
            self._cache_metadata = cache_data['metadata']
            return True

        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False

    def save_cache(self, frameworks_file: Optional[str] = None) -> bool:
        """
        Save embeddings to cache.

        Args:
            frameworks_file: Path to frameworks file for timestamp

        Returns:
            True if cache was saved, False otherwise
        """
        try:
            # Ensure cache directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            metadata = {
                'model': self.model,
                'dimensions': self.dimensions,
                'frameworks_timestamp': get_file_timestamp(frameworks_file),
                'num_embeddings': len(self.embeddings),
                'token_usage': self.token_usage
            }

            cache_data = {
                'embeddings': self.embeddings,
                'metadata': metadata
            }

            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            self._cache_metadata = metadata
            return True

        except Exception as e:
            print(f"Failed to save cache: {e}")
            return False

    def get_or_create_embeddings(
        self,
        df: pd.DataFrame,
        frameworks_file: Optional[str] = None,
        show_progress: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        Load embeddings from cache or create them.

        Args:
            df: DataFrame with frameworks
            frameworks_file: Path to frameworks file
            show_progress: Whether to show progress during embedding

        Returns:
            Dict mapping framework IDs to embeddings
        """
        # Check if caching is enabled
        cache_enabled = os.getenv('ENABLE_EMBEDDING_CACHE', 'true').lower() == 'true'

        if cache_enabled and self.load_cache(frameworks_file):
            # Verify all IDs are in cache
            missing_ids = set(df['id'].tolist()) - set(self.embeddings.keys())

            if not missing_ids:
                return self.embeddings

            # Embed only missing frameworks
            missing_df = df[df['id'].isin(missing_ids)]
            if show_progress:
                print(f"Embedding {len(missing_ids)} new frameworks...")

            new_embeddings = self.embed_frameworks(missing_df, show_progress=show_progress)
            self.embeddings.update(new_embeddings)
            self.save_cache(frameworks_file)

            return self.embeddings

        # Create all embeddings
        embeddings = self.embed_frameworks(df, show_progress=show_progress)

        if cache_enabled:
            self.save_cache(frameworks_file)

        return embeddings

    def get_embedding(self, framework_id: int) -> Optional[np.ndarray]:
        """
        Get embedding for a specific framework.

        Args:
            framework_id: Framework ID

        Returns:
            Embedding array or None if not found
        """
        return self.embeddings.get(framework_id)

    def get_all_embeddings_matrix(self, ids: List[int]) -> Tuple[np.ndarray, List[int]]:
        """
        Get embeddings as a matrix for efficient similarity computation.

        Args:
            ids: List of framework IDs to include

        Returns:
            Tuple of (embedding matrix, list of IDs in same order)
        """
        valid_ids = [id for id in ids if id in self.embeddings]
        embeddings_list = [self.embeddings[id] for id in valid_ids]

        if not embeddings_list:
            return np.array([]), []

        return np.vstack(embeddings_list), valid_ids

    def get_token_usage(self) -> int:
        """
        Get total token usage for embedding operations.

        Returns:
            Total tokens used
        """
        return self.token_usage

    def get_cost_estimate(self, price_per_million: float = 0.02) -> float:
        """
        Estimate cost of embedding operations.

        Args:
            price_per_million: Price per million tokens (default for text-embedding-3-small)

        Returns:
            Estimated cost in dollars
        """
        return (self.token_usage / 1_000_000) * price_per_million

    def clear_cache(self) -> bool:
        """
        Delete the cache file.

        Returns:
            True if cache was deleted, False otherwise
        """
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            self.embeddings = {}
            self._cache_metadata = {}
            return True
        except Exception as e:
            print(f"Failed to clear cache: {e}")
            return False
