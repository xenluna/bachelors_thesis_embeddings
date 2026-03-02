# semantic_search.py
import json
from typing import Dict, Any, List

import numpy as np

from vector_index import ProductIndex, Embedder


DEFAULT_INDEX_DIR = "product_index"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def semantic_search(query: str, k: int = 5,
                    index_dir: str = DEFAULT_INDEX_DIR,
                    model_name: str = DEFAULT_MODEL_NAME) -> str:
    """
    Returns results as a JSON string (good for tool-calling).
    """
    embedder = Embedder(model_name=model_name)
    pindex = ProductIndex.load(index_dir)

    q_vec = embedder.embed_query(query)
    results = pindex.search(q_vec, k=k)

    out: List[Dict[str, Any]] = []
    for score, p in results:
        out.append({
            "score": round(score, 4),
            "id": p.get("id"),
            "title": p.get("title"),
            "category": p.get("category"),
            "ingredients": p.get("ingredients"),
        })

    return json.dumps({"query": query, "top_k": k, "results": out}, ensure_ascii=False)


def cli_loop() -> None:
    while True:
        q = input("\nQuery (enter to exit): ").strip()
        if not q:
            break
        print(semantic_search(q, k=5))


if __name__ == "__main__":
    cli_loop()