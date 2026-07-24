"""Điền embedding cho các dòng math_question_bank đang NULL.

Chạy ở nơi có endpoint embedding (GPU pod) — cấu hình:
  EMBEDDING_BACKEND=remote  EMBEDDING_API_URL=https://<pod>-8001.proxy.runpod.net/v1
hoặc EMBEDDING_BACKEND=local nếu máy đủ khoẻ để nạp bge-m3.

  python -m tools.backfill_embeddings                 # điền hết dòng NULL
  python -m tools.backfill_embeddings --limit 100     # thử 100 dòng
  python -m tools.backfill_embeddings --batch 64
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AI_service import db  # noqa: E402
from AI_service.embeddings import build_question_text, embed_texts  # noqa: E402


def run(limit: int | None, batch: int) -> None:
    t0 = time.time()
    with db.connect() as c:
        pending = int(c.execute("SELECT count(*) FROM math_question_bank WHERE embedding IS NULL").fetchone()[0])
    print(f"Dong can embed: {pending}" + (f" (gioi han {limit})" if limit else ""))

    done = 0
    while True:
        take = batch if limit is None else min(batch, limit - done)
        if take <= 0:
            break
        with db.connect() as c:
            rows = c.execute(
                "SELECT id, document FROM math_question_bank WHERE embedding IS NULL LIMIT %s", [take]
            ).fetchall()
        if not rows:
            break

        ids = [r[0] for r in rows]
        vectors = embed_texts([build_question_text(r[1]) for r in rows])
        with db.connect() as c:
            cur = c.cursor()
            cur.executemany(
                "UPDATE math_question_bank SET embedding = %s::vector, updated_at = now() WHERE id = %s",
                [(db._vec(v), qid) for qid, v in zip(ids, vectors)],
            )
        done += len(rows)
        print(f"  embed {done}  ({time.time()-t0:.0f}s)")

    print(f"\nXONG backfill: {done} dong, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    run(args.limit, args.batch)
