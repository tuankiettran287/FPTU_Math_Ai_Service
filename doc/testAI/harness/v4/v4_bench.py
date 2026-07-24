# -*- coding: utf-8 -*-
"""Chạy bộ test V4 (đánh giá năng lực) qua endpoint /api/v1/competency/grade của
AI service ĐANG phục vụ 1 model, ghi RAW response ra JSONL. Chấm để sau (v4_grade.py).

Mỗi item test = câu hỏi + bài làm SV viết sẵn + nhãn expected. Ta gửi cho AI chấm y
như production (enrich skill_targets bằng topics từ framework + gửi đủ 10 GMC dim —
đúng những gì BE gửi), rồi lưu nguyên response.

Chạy TRÊN POD (hit localhost:8080, không qua proxy). Gọi ĐỒNG THỜI nhiều request để
vLLM continuous-batching → rút ngắn wall-time; chất lượng mỗi request vẫn độc lập.

  python3 v4_bench.py --model r1_32b --base http://localhost:8080 \
      --testdir /workspace/testset_v4 --out /workspace/logs/v4 --concurrency 8
"""
from __future__ import annotations
import argparse, json, os, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

SUBJECTS = ["MAE101", "MAD101", "MAS291"]


def http_json(base, path, body=None, timeout=1200):
    url = base + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "curl/8.7.1"},
                                 method="POST" if data else "GET")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), round(time.time() - t0, 1)


def load_framework(base, subject):
    """skill_id → topics(list), và list GMC dims (id+name) — đúng như BE build."""
    fw, _ = http_json(base, f"/api/v1/competency/framework?subject={subject}")
    topics = {}
    for clo in (fw.get("subject", {}).get("clos") or []):
        for s in (clo.get("skills") or []):
            topics[s.get("skill_id")] = s.get("topics") or []
    dims = [{"dimension_id": d.get("dimension_id"), "name": d.get("name_vi") or d.get("name_en")}
            for d in (fw.get("gmc", {}).get("dimensions") or [])]
    return topics, dims


def build_grade_body(item, topics_map, gmc_dims):
    q = item["question"]
    skills = []
    for t in item["skill_targets"]:
        sid = t.get("skill_id")
        skills.append({
            "skill_id": sid,
            "name": t.get("name") or sid,
            "topics": ", ".join(topics_map.get(sid, [])),  # phạm vi syllabus cho scope
        })
    return {
        "question": {
            "id": q.get("id"),
            "question_text": q.get("question_text"),
            "standard_solution": q.get("standard_solution"),
            "final_answer": q.get("final_answer"),
            "accepted_methods": q.get("accepted_methods") or [],
            "grading_notes": q.get("grading_notes") or "",
            "points": q.get("points") or 10,
        },
        "student_answer": item.get("student_answer") or "",
        "skill_targets": skills,
        "gmc_targets": gmc_dims,
        "hint_events": [
            {"level": h.get("level"), "hint_text": h.get("hint_text") or "",
             "revealed": h.get("revealed") or {},
             "draft_at_request": h.get("draft_at_request") or ""}
            for h in (item.get("hint_events") or [])
        ],
        "language": "vi",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="giới hạn số câu/môn (0=tất cả)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"graded_{args.model}.jsonl")

    # Đã chạy tới đâu (resume): bỏ qua test_id đã có trong file.
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["test_id"])
            except Exception:
                pass

    fw_cache = {}
    jobs = []
    for subj in SUBJECTS:
        path = os.path.join(args.testdir, f"FPT_V4_Competency_Test_Set_45_{subj}.json")
        if not os.path.exists(path):
            print(f"!! thiếu {path}", flush=True)
            continue
        data = json.load(open(path, encoding="utf-8"))
        items = data["items"] if isinstance(data, dict) else data
        if args.limit:
            items = items[:args.limit]
        if subj not in fw_cache:
            fw_cache[subj] = load_framework(args.base, subj)
        topics_map, gmc_dims = fw_cache[subj]
        for it in items:
            if it["test_id"] in done:
                continue
            jobs.append((subj, it, topics_map, gmc_dims))

    print(f"[{args.model}] cần chạy {len(jobs)} câu (đã có {len(done)}), concurrency={args.concurrency}", flush=True)
    fout = open(out_path, "a", encoding="utf-8")
    t_start = time.time()
    n_done = 0

    def run_one(job):
        subj, it, topics_map, gmc_dims = job
        body = build_grade_body(it, topics_map, gmc_dims)
        try:
            resp, dt = http_json(args.base, "/api/v1/competency/grade", body, timeout=1200)
            return {"test_id": it["test_id"], "subject": subj, "scenario": it["scenario"],
                    "difficulty": it.get("difficulty"), "expected": it["expected"],
                    "latency_s": dt, "ok": True, "response": resp}
        except Exception as e:
            return {"test_id": it["test_id"], "subject": subj, "scenario": it["scenario"],
                    "expected": it["expected"], "ok": False, "error": str(e)[:400]}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 5 == 0 or not rec.get("ok"):
                el = time.time() - t_start
                rate = n_done / el if el else 0
                eta = (len(jobs) - n_done) / rate if rate else 0
                print(f"[{args.model}] {n_done}/{len(jobs)}  "
                      f"{rec['test_id']} {'OK' if rec.get('ok') else 'ERR '+rec.get('error','')[:60]}  "
                      f"ETA {eta/60:.0f}m", flush=True)
    fout.close()
    print(f"[{args.model}] XONG {n_done} câu trong {(time.time()-t_start)/60:.1f} phút → {out_path}", flush=True)


if __name__ == "__main__":
    main()
