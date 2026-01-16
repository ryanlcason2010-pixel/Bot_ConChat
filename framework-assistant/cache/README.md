# Cache Directory

This directory stores cached embeddings to speed up subsequent runs.

## Files

- `embeddings_cache.pkl`: Pickle file containing pre-computed embeddings for all frameworks.

## Cache Invalidation

The cache is automatically invalidated when:
1. The `frameworks.xlsx` file is modified (based on file timestamp)
2. The embedding model is changed in `.env`
3. The cache file is manually deleted

## Manual Cache Clear

To clear the cache and force re-embedding:

```bash
rm cache/embeddings_cache.pkl
```

Then restart the application. The embeddings will be regenerated.

## Storage

The cache file is typically 5-20 MB depending on the number of frameworks.
It is excluded from git via `.gitignore`.
