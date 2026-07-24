"""Xuất RAG corpus THẬT của hệ thống từ Postgres local -> file, để upload lên pod.

Vì sao export ra file thay vì cho pod nối thẳng DB:
  - Benchmark phải tái lập được: cùng corpus, cùng vector, không phụ thuộc tunnel.
  - Độ trễ mạng DB sẽ làm bẩn số đo TTFT/latency của model.
  - Embedding đã có sẵn trong DB (bge-m3 1024 chiều) -> KHÔNG cần embed lại,
    giữ đúng vector mà production đang dùng.

Nguồn (đúng những gì production truy hồi):
  A. document_chunks  — lý thuyết, hiện CHỈ có MAD101 (405 chunk / 8 tài liệu)
  B. math_question_bank — 14.5k bài đã giải, đủ MAE/MAD/MAS
"""
import json
import sys
from pathlib import Path

import numpy as np
import psycopg

DB = "postgresql://fptu_admin:7835be287eb6d4467d1f93085f63c3ede81ec21e42e5caa8@127.0.0.1:15432/fptu_mathai_final"
OUT = Path(__file__).parent / "corpus"
OUT.mkdir(exist_ok=True)


def parse_vec(v):
    """pgvector trả về str '[0.1,0.2,...]' hoặc list, tuỳ driver."""
    if v is None:
        return None
    if isinstance(v, str):
        return np.fromstring(v.strip("[]"), sep=",", dtype=np.float32)
    return np.asarray(v, dtype=np.float32)


def dump(name, rows, text_fn):
    """rows -> {name}.jsonl (metadata) + {name}.npy (ma trận embedding đã chuẩn hoá)."""
    meta, vecs = [], []
    for r in rows:
        v = parse_vec(r["embedding"])
        if v is None or v.shape[0] != 1024:
            continue
        m = dict(r)
        m.pop("embedding", None)
        m["_text"] = text_fn(r)
        meta.append(m)
        vecs.append(v)
    M = np.vstack(vecs).astype(np.float32)
    # Chuẩn hoá L2 sẵn -> retrieval chỉ còn 1 phép nhân ma trận (cosine = dot).
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    np.save(OUT / f"{name}.npy", M)
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False, default=str) + "\n")
    print(f"  {name}: {len(meta)} mục, vector {M.shape}, "
          f"{(OUT / f'{name}.npy').stat().st_size / 1e6:.1f}MB")
    return len(meta)


def chunk_text(r):
    head = " · ".join(x for x in [r.get("title"), r.get("chapter"), r.get("section")] if x)
    return f"{head}\n{r.get('content') or ''}".strip()


def qb_text(r):
    """Ghép giống build_question_text() của production, CỘNG THÊM lời giải & lỗi thường gặp.

    Production embed = header + đề + đáp án cuối (dùng để TÌM câu tương tự).
    Ở đây phần embed giữ nguyên vector cũ trong DB; chỉ phần TEXT hiển thị cho LLM
    mới thêm các bước giải + common_mistakes — vì đó mới là thứ giúp model giải/chấm.
    """
    q = r.get("question") or {}
    sol = r.get("solution") or {}
    if isinstance(q, str):
        q = json.loads(q)
    if isinstance(sol, str):
        sol = json.loads(sol)
    head = " · ".join(str(x) for x in [r.get("subject"), r.get("chapter"), r.get("topic")] if x)
    parts = [head, f"Đề: {q.get('text') or ''}"]
    if q.get("latex"):
        parts.append(f"LaTeX: {q['latex']}")
    steps = sol.get("steps") or []
    if steps:
        sl = []
        for s in steps[:8]:
            sl.append(s if isinstance(s, str) else
                      " ".join(str(s.get(k, "")) for k in ("description", "expression", "text") if s.get(k)))
        parts.append("Các bước giải: " + " | ".join(x for x in sl if x))
    if sol.get("final_answer"):
        parts.append(f"Đáp án: {sol['final_answer']}")
    cu = r.get("concepts_used")
    if cu:
        parts.append(f"Khái niệm dùng: {cu if isinstance(cu, str) else ', '.join(map(str, cu))}")
    cm = r.get("common_mistakes")
    if cm:
        s = cm if isinstance(cm, str) else "; ".join(
            (x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)) for x in cm[:4])
        parts.append(f"Lỗi thường gặp: {s}")
    return "\n".join(p for p in parts if p and p.strip())[:4000]


def main():
    with psycopg.connect(DB, connect_timeout=15) as c:
        with c.cursor(row_factory=psycopg.rows.dict_row) as k:
            print("Xuất document_chunks (lý thuyết)...")
            k.execute("""SELECT c.id, c.document_id, c.ordinal, c.content, c.embedding,
                                d.title, d.subject, d.course
                         FROM document_chunks c JOIN documents d ON d.id = c.document_id
                         WHERE c.embedding IS NOT NULL""")
            rows = k.fetchall()
            for r in rows:
                r.setdefault("chapter", None)
                r.setdefault("section", None)
            n_chunks = dump("chunks", rows, chunk_text)

            print("Xuất math_question_bank (bài giải mẫu)...")
            k.execute("""SELECT id, subject, course, chapter, topic, subtopic, difficulty,
                                question_type, question, solution, concepts_used,
                                common_mistakes, embedding
                         FROM math_question_bank WHERE embedding IS NOT NULL""")
            n_qb = dump("qb", k.fetchall(), qb_text)

    # Phân bố môn -> để báo cáo độ phủ corpus (bằng chứng cho slide phân tích lỗi).
    import collections
    cov = {}
    for name in ("chunks", "qb"):
        cnt = collections.Counter()
        with open(OUT / f"{name}.jsonl", encoding="utf-8") as f:
            for line in f:
                cnt[(json.loads(line).get("subject") or "?")] += 1
        cov[name] = dict(cnt)
    (OUT / "coverage.json").write_text(
        json.dumps({"chunks": cov["chunks"], "qb": cov["qb"],
                    "n_chunks": n_chunks, "n_qb": n_qb}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("\nĐộ phủ corpus theo môn:")
    print(json.dumps(cov, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
