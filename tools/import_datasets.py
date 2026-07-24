"""Import dataset MAD/MAE/MAS + Data_Bank (1).json vào math_question_bank.

- Parser 'salvage': quét từng object top-level, tự sửa lỗi JSON hay gặp
  (key assest_format lặp, thiếu dấu phẩy giữa các trường/phần tử), bỏ qua
  bản ghi không cứu được thay vì hỏng cả file.
- Dedupe theo id (Data_Bank xử lý trước vì sạch, rồi bổ sung folder-only).
- Sinh embedding bge-m3 theo batch rồi upsert.

MẶC ĐỊNH import với embedding = NULL (máy dev không có GPU, không load model).
Điền embedding sau bằng `tools/backfill_embeddings.py` khi có endpoint embedding
trên GPU pod (hoặc thêm cờ --embed nếu EMBEDDING_BACKEND đã trỏ tới endpoint remote).

Chạy:
  python -m tools.import_datasets --dry-run          # chỉ đếm, không ghi DB
  python -m tools.import_datasets                     # import (embedding = NULL)
  python -m tools.import_datasets --embed            # import + sinh embedding (cần endpoint)
  python -m tools.import_datasets --limit 20          # thử nhanh 20 câu
"""
import argparse
import glob
import json
import os
import re
import sys
import time

# cho phép chạy như script trực tiếp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AI_service import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── parser salvage ────────────────────────────────────────────────────────────
def _iter_top_objects(raw: str):
    """Yield từng chuỗi object top-level '{...}' bên trong mảng ngoài cùng."""
    depth = 0
    in_str = False
    esc = False
    start = -1
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                yield raw[start : i + 1]
                start = -1


_DUP_ASSEST = re.compile(r'("assest_format"\s*:\s*"[^"]*")\s*,?\s*"assest_format"\s*:\s*"[^"]*"')
_MISSING_COMMA = re.compile(r'((?:"[^"]*"|\}|\]|\d|true|false|null))(\s*\n\s*)(")')


def _repair(obj: str) -> str:
    obj = _DUP_ASSEST.sub(r"\1", obj)              # gộp key assest_format lặp
    prev = None
    while prev != obj:                              # thêm phẩy thiếu (lặp tới ổn định)
        prev = obj
        obj = _MISSING_COMMA.sub(r"\1,\2\3", obj)
    return obj


def robust_records(path: str) -> tuple[list[dict], int]:
    raw = open(path, encoding="utf-8-sig").read()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data, 0
    except json.JSONDecodeError:
        pass
    records: list[dict] = []
    skipped = 0
    for chunk in _iter_top_objects(raw):
        for candidate in (chunk, _repair(chunk)):
            try:
                records.append(json.loads(candidate))
                break
            except json.JSONDecodeError:
                continue
        else:
            skipped += 1
    return records, skipped


# ── thu thập ──────────────────────────────────────────────────────────────────
def collect() -> list[dict]:
    seen: set[str] = set()
    ordered: list[dict] = []

    def add(records: list[dict], label: str) -> int:
        added = 0
        for r in records:
            rid = r.get("id")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            ordered.append(r)
            added += 1
        print(f"  [{label}] +{added} moi (tong unique={len(ordered)})")
        return added

    # Data_Bank trước (sạch)
    dbank = os.path.join(ROOT, "Data_Bank (1).json")
    if os.path.exists(dbank):
        recs, skip = robust_records(dbank)
        print(f"Data_Bank: {len(recs)} ban ghi, skip {skip}")
        add(recs, "Data_Bank")

    # rồi các folder
    for sub in ("MAD", "MAE", "MAS"):
        for f in sorted(glob.glob(os.path.join(ROOT, sub, "*.json"))):
            recs, skip = robust_records(f)
            if skip:
                print(f"  {os.path.basename(f)}: {len(recs)} ban ghi, skip {skip} (hong)")
            add(recs, os.path.basename(f)[:28])
    return ordered


# ── import ────────────────────────────────────────────────────────────────────
def run(dry_run: bool, limit: int | None, batch: int, embed: bool) -> None:
    t0 = time.time()
    records = collect()
    if limit:
        records = records[:limit]
    print(f"\nTong so cau se import: {len(records)}  (embed={embed})")
    if dry_run:
        print("(dry-run: khong ghi DB, khong embed)")
        return

    build_text = embed_fn = None
    if embed:
        from AI_service.embeddings import build_question_text, embed_texts
        build_text, embed_fn = build_question_text, embed_texts

    db.init_db()
    total = 0
    for i in range(0, len(records), batch):
        part = records[i : i + batch]
        if embed:
            vectors = embed_fn([build_text(r) for r in part])
            items = list(zip(part, vectors))
        else:
            items = [(r, None) for r in part]
        with db.connect() as conn:
            db.upsert_questions_bulk(items, conn=conn)
        total += len(part)
        print(f"  upsert {total}/{len(records)}  ({time.time()-t0:.0f}s)")
    print(f"\nXONG: {total} cau, {time.time()-t0:.0f}s. Tong trong DB: {db.count_questions()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--embed", action="store_true", help="sinh embedding (cần endpoint remote)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    run(args.dry_run, args.limit, args.batch, args.embed)
