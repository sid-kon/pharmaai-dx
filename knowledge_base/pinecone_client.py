import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialise the embedding model once at module level
_embedder = None


def _get_embedder() -> SentenceTransformer:
    """Lazy-load the sentence transformer model."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def init_pinecone():
    """
    Initialise the Pinecone client and return the index object.
    Reads PINECONE_API_KEY and PINECONE_INDEX_NAME from environment.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "pharmaai-dx")

    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set in environment variables.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index


def query_pinecone(
    query: str,
    dimension_filter: str = None,
    top_k: int = 5,
) -> list[str]:
    """
    Embed a query string and retrieve the top_k matching chunks
    from Pinecone.

    Args:
        query:            The search query string.
        dimension_filter: Optional. If provided, filters results to
                          chunks whose metadata "dimension" field
                          matches this string.
        top_k:            Number of results to return (default 5).

    Returns:
        A list of text strings from the matched chunks.
        Returns an empty list on any failure.
    """
    try:
        index = init_pinecone()
        embedder = _get_embedder()

        # Embed the query
        query_vector = embedder.encode(query).tolist()

        # Build optional metadata filter
        filter_dict = None
        if dimension_filter:
            filter_dict = {"dimension": {"$eq": dimension_filter}}

        # Query Pinecone
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
        )

        # Extract text from metadata
        chunks = []
        for match in results.get("matches", []):
            text = match.get("metadata", {}).get("text", "")
            if text:
                chunks.append(text)

        return chunks

    except Exception as e:
        print(f"[pinecone_client] Query failed: {e}")
        return []

