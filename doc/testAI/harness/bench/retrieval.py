"""RAG retrieval — bản sao trung thực của cách production truy hồi, nhưng chạy
trên chỉ mục numpy trong RAM thay vì Postgres.

Vì sao không gọi thẳng Postgres:
  Mỗi truy vấn DB qua tunnel tốn 50-200ms và dao động -> làm bẩn số đo TTFT.
  Embedding trong DB được xuất nguyên vẹn ra .npy nên KẾT QUẢ TRUY HỒI GIỐNG HỆT
  (cùng vector, cùng độ đo cosine), chỉ khác chỗ tính.

Hai nguồn — đúng như production:
  - chunks: lý thuyết (document_chunks)      -> lọc theo môn
  - qb:     bài đã giải (math_question_bank) -> lọc theo môn
Truy vấn được embed bằng CHÍNH bge-m3 đang phục vụ trên pod.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .client import embed

# Ngân hàng câu hỏi dùng mã ngắn (MAE/MAD/MAS); tài liệu & bộ test dùng mã đầy đủ.
SUBJ_SHORT = {"MAE101": "MAE", "MAD101": "MAD", "MAS291": "MAS"}


@dataclass
class Hit:
    source: str          # 'theory' | 'exemplar'
    score: float         # cosine
    text: str
    ref: str             # nhãn nguồn để trích dẫn + để chấm groundedness
    subject: str | None


class Index:
    def __init__(self, corpus_dir: Path):
        self.dir = Path(corpus_dir)
        self.chunks_meta = self._load_jsonl("chunks.jsonl")
        self.chunks_vec = np.load(self.dir / "chunks.npy")
        self.qb_meta = self._load_jsonl("qb.jsonl")
        self.qb_vec = np.load(self.dir / "qb.npy")
        # Mảng mã môn để lọc bằng vector hoá (nhanh hơn lặp Python).
        self.chunks_subj = np.array([(m.get("subject") or "") for m in self.chunks_meta])
        self.qb_subj = np.array([(m.get("subject") or "") for m in self.qb_meta])

    def _load_jsonl(self, name: str) -> list[dict]:
        with open(self.dir / name, encoding="utf-8") as f:
            return [json.loads(l) for l in f]

    def embed_query(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(embed(texts, port=config.EMBED_PORT, model=config.EMBED_MODEL),
                       dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    def _top(self, qv, mat, meta, subj_arr, subject_full, k, source) -> list[Hit]:
        if mat.shape[0] == 0 or k <= 0:
            return []
        sims = mat @ qv                       # vector đã chuẩn hoá -> dot = cosine
        want = {subject_full, SUBJ_SHORT.get(subject_full, "")} - {""}
        if want:
            mask = np.isin(subj_arr, list(want))
            if mask.any():
                sims = np.where(mask, sims, -2.0)   # loại môn khác, không xoá mảng
        idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        out = []
        for i in idx:
            if sims[i] < -1:
                continue
            m = meta[int(i)]
            ref = (m.get("title") or m.get("topic") or m.get("id") or source)
            out.append(Hit(source=source, score=float(sims[i]), text=m["_text"],
                           ref=f"{source}:{ref}", subject=m.get("subject")))
        return out

    def retrieve(self, query: str, subject_full: str, *,
                 k_theory: int | None = None, k_exemplar: int | None = None,
                 qv: np.ndarray | None = None) -> list[Hit]:
        kt = config.RETRIEVAL_K_CHUNKS if k_theory is None else k_theory
        ke = config.RETRIEVAL_K_QB if k_exemplar is None else k_exemplar
        if qv is None:
            qv = self.embed_query([query])[0]
        hits = self._top(qv, self.chunks_vec, self.chunks_meta, self.chunks_subj,
                         subject_full, kt, "theory")
        hits += self._top(qv, self.qb_vec, self.qb_meta, self.qb_subj,
                          subject_full, ke, "exemplar")
        return hits


def format_context(hits: list[Hit]) -> str:
    """Ghép ngữ cảnh cho prompt, có đánh số [1][2].. để model trích dẫn được.

    Đánh số là bắt buộc: chấm Groundedness cần biết model dựa vào đoạn NÀO.
    """
    parts = []
    for i, h in enumerate(hits, 1):
        kind = "LÝ THUYẾT" if h.source == "theory" else "BÀI GIẢI MẪU"
        parts.append(f"[{i}] ({kind} · {h.ref} · độ tương đồng {h.score:.3f})\n{h.text}")
    return "\n\n".join(parts)
