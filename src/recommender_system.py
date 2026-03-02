# recommender_demo.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from vector_index import ProductIndex

import os
from vector_index import build_index

if not os.path.exists("product_index/faiss.index"):
    build_index("mock_data_products.jsonl", out_dir="product_index")



@dataclass
class Recommendation:
    product_id: str
    title: str
    category: str
    score_cf: float
    score_embed: float
    score_final: float


def load_users_jsonl(path: str) -> List[dict]:
    users = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                users.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}")
    return users


def build_interactions(users: List[dict]) -> Dict[str, Dict[str, float]]:
    """
    Convert JSONL users into:
      interactions[user_id][product_id] = summed_weight
    """
    interactions: Dict[str, Dict[str, float]] = {}
    for u in users:
        uid = str(u["user_id"])
        interactions[uid] = {}
        for ev in u.get("interactions", []):
            pid = str(ev["product_id"])
            w = float(ev.get("weight", 1.0))
            interactions[uid][pid] = interactions[uid].get(pid, 0.0) + w
    return interactions


class HybridRecommender:
    """
    Item-item collaborative filtering (implicit) + embedding re-ranking.
    Uses FAISS vectors already stored in your ProductIndex.
    """

    def __init__(self, index_dir: str = "product_index"):
        self.pindex = ProductIndex.load(index_dir)

        # product_id -> faiss idx
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_payload: Dict[int, dict] = {}

        for i, p in enumerate(self.pindex.payloads):
            pid = p.get("id")
            if pid is None:
                continue
            pid = str(pid)
            self.id_to_idx[pid] = i
            self.idx_to_payload[i] = p

        self.item_ids: List[str] = list(self.id_to_idx.keys())
        self.item_id_to_col: Dict[str, int] = {pid: j for j, pid in enumerate(self.item_ids)}

        self.item_sim: np.ndarray | None = None
        self.interactions: Dict[str, Dict[str, float]] | None = None

    def _item_vec(self, pid: str) -> np.ndarray:
        idx = self.id_to_idx[pid]
        v = self.pindex.index.reconstruct(idx)
        v = np.asarray(v, dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v

    def fit(self, interactions: Dict[str, Dict[str, float]]) -> None:
        """
        Build item-item cosine similarity using dense user-item matrix.
        Fine for mock data / small scale.
        """
        self.interactions = interactions

        user_ids = list(interactions.keys())
        U = len(user_ids)
        I = len(self.item_ids)

        X = np.zeros((U, I), dtype=np.float32)

        for ui, uid in enumerate(user_ids):
            for pid, w in interactions[uid].items():
                if pid in self.item_id_to_col:
                    X[ui, self.item_id_to_col[pid]] = float(w)

        XtX = X.T @ X
        norms = np.sqrt(np.diag(XtX)) + 1e-12
        sim = XtX / norms[:, None] / norms[None, :]

        np.fill_diagonal(sim, 0.0)
        self.item_sim = sim

    def recommend(
        self,
        user_id: str,
        top_k: int = 5,
        alpha: float = 0.80,
        candidate_pool: int = 30
    ) -> List[Recommendation]:
        if self.item_sim is None or self.interactions is None:
            raise RuntimeError("Call fit(interactions) before recommend().")

        user_hist = self.interactions.get(user_id, {})
        if not user_hist:
            return self._cold_start(top_k)

        # CF scoring: sum similarities to interacted items
        scores_cf = np.zeros((len(self.item_ids),), dtype=np.float32)

        for pid_hist, w in user_hist.items():
            if pid_hist not in self.item_id_to_col:
                continue
            j = self.item_id_to_col[pid_hist]
            scores_cf += self.item_sim[:, j] * float(w)

        # exclude already seen items
        for pid_hist in user_hist.keys():
            if pid_hist in self.item_id_to_col:
                scores_cf[self.item_id_to_col[pid_hist]] = -1e9

        cand_cols = np.argsort(-scores_cf)[:candidate_pool]
        cand_pids = [self.item_ids[c] for c in cand_cols if scores_cf[c] > -1e8]

        # user profile embedding (weighted mean)
        profile = self._user_profile_vec(user_hist)

        recs: List[Recommendation] = []
        for pid in cand_pids:
            payload = self.idx_to_payload[self.id_to_idx[pid]]
            v = self._item_vec(pid)

            s_cf = float(scores_cf[self.item_id_to_col[pid]])
            s_embed = float(np.dot(profile, v))
            s_final = alpha * s_cf + (1.0 - alpha) * s_embed

            recs.append(
                Recommendation(
                    product_id=pid,
                    title=str(payload.get("title", "")),
                    category=str(payload.get("category", "")),
                    score_cf=s_cf,
                    score_embed=s_embed,
                    score_final=s_final,
                )
            )

        recs.sort(key=lambda r: r.score_final, reverse=True)
        return recs[:top_k]

    def _user_profile_vec(self, user_hist: Dict[str, float]) -> np.ndarray:
        vecs = []
        weights = []
        for pid, w in user_hist.items():
            if pid in self.id_to_idx:
                vecs.append(self._item_vec(pid))
                weights.append(float(w))

        if not vecs:
            return np.zeros((self.pindex.dim,), dtype=np.float32)

        V = np.stack(vecs, axis=0)
        W = np.asarray(weights, dtype=np.float32)
        prof = (V * W[:, None]).sum(axis=0) / (W.sum() + 1e-12)
        prof /= (np.linalg.norm(prof) + 1e-12)
        return prof.astype(np.float32)

    def _cold_start(self, top_k: int) -> List[Recommendation]:
        recs: List[Recommendation] = []
        for pid in self.item_ids[:top_k]:
            payload = self.idx_to_payload[self.id_to_idx[pid]]
            recs.append(
                Recommendation(
                    product_id=pid,
                    title=str(payload.get("title", "")),
                    category=str(payload.get("category", "")),
                    score_cf=0.0,
                    score_embed=0.0,
                    score_final=0.0,
                )
            )
        return recs


def print_offer(user_id: str, segment: str, recs: List[Recommendation]) -> None:
    print(f"\n=== Personalisiertes Angebot für {user_id} (Segment: {segment}) ===")
    for i, r in enumerate(recs, start=1):
        print(f"{i}. {r.title} [{r.category}]  final={r.score_final:.4f}  (cf={r.score_cf:.4f}, emb={r.score_embed:.4f})")


def main() -> None:
    print("RUNNING:", os.path.abspath(__file__))
    print("CWD:", os.getcwd())

    INDEX_DIR = "product_index"
    USERS_JSONL = "mock_data_customers.jsonl"

    print("Users file exists?", os.path.exists(USERS_JSONL))
    if os.path.exists(USERS_JSONL):
        print("Users file size:", os.path.getsize(USERS_JSONL), "bytes")

    users = load_users_jsonl(USERS_JSONL)
    print("Users loaded:", len(users))
    INDEX_DIR = "product_index"
    USERS_JSONL = "mock_data_customers.jsonl"

    users = load_users_jsonl(USERS_JSONL)
    interactions = build_interactions(users)

    rec = HybridRecommender(index_dir=INDEX_DIR)
    rec.fit(interactions)

    for u in users:
        uid = str(u["user_id"])
        segment = str(u.get("segment", ""))
        recs = rec.recommend(uid, top_k=5, alpha=0.80, candidate_pool=30)
        print_offer(uid, segment, recs)


if __name__ == "__main__":
    main()