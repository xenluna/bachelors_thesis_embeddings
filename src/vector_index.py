# vector_index.py
#setus up the vector index once
#works and finished

import json
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}")
    return items


def product_to_text(p: Dict) -> str:
    title = (p.get("title") or "").strip()
    category = (p.get("category") or "").strip()

    ingr = p.get("ingredients") or []
    if isinstance(ingr, list):
        ingredients = ", ".join([str(x).strip() for x in ingr if str(x).strip()])
    else:
        ingredients = str(ingr).strip()

    return f"Titel: {title}\nKategorie: {category}\nZutaten: {ingredients}"


class ProductIndex:
    """
    FAISS IndexFlatIP + Cosine similarity via L2-normalization.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.payloads: List[Dict] = []

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        x = x.astype("float32")
        faiss.normalize_L2(x)
        return x

    def add(self, vectors: np.ndarray, payloads: List[Dict]) -> None:
        assert vectors.shape[0] == len(payloads)
        vectors = self._norm(vectors)
        self.index.add(vectors)
        self.payloads.extend(payloads)

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[float, Dict]]:
        q = query_vec.reshape(1, -1)
        q = self._norm(q)
        scores, idxs = self.index.search(q, k)

        out: List[Tuple[float, Dict]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append((float(score), self.payloads[int(idx)]))
        return out

    def save(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "faiss.index"))
        with open(os.path.join(folder, "payloads.json"), "w", encoding="utf-8") as f:
            json.dump(self.payloads, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, folder: str) -> "ProductIndex":
        idx = faiss.read_index(os.path.join(folder, "faiss.index"))
        with open(os.path.join(folder, "payloads.json"), "r", encoding="utf-8") as f:
            payloads = json.load(f)
        obj = cls(dim=idx.d)
        obj.index = idx
        obj.payloads = payloads
        return obj


class Embedder:
    """
    Separate class so you can swap BERT/SBERT models later without touching indexing/search logic.
    """
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query])
        return np.asarray(vec, dtype="float32")[0]


def build_index(jsonl_path: str, out_dir: str = "product_index",
                model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
    products = read_jsonl(jsonl_path)
    embedder = Embedder(model_name=model_name)

    texts = [product_to_text(p) for p in products]
    vecs = embedder.embed_texts(texts, batch_size=32)

    pindex = ProductIndex(dim=vecs.shape[1])

    payloads = []
    for p, t in zip(products, texts):
        payloads.append({
            "id": p.get("id"),
            "title": p.get("title"),
            "category": p.get("category"),
            "ingredients": p.get("ingredients"),
            "embedded_text": t
        })

    pindex.add(vecs, payloads)
    pindex.save(out_dir)
    print(f"Saved index to: {out_dir}")

if __name__ == "__main__":
    build_index("mock_data_products.jsonl", out_dir="product_index")