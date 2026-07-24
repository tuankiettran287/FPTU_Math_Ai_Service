# -*- coding: utf-8 -*-
"""Chấm kết quả V4 BẰNG CODE (không LLM giám khảo) — so response của AI với nhãn
expected trong test set. Đọc mọi graded_*.jsonl trong 1 thư mục, xuất summary JSON
+ bảng markdown per-model/per-scenario.

  python3 v4_grade.py --dir results/v4 --out results/v4/summary.json

6 phép kiểm mỗi câu (chỉ tính phép nào áp dụng được cho câu đó):
  answers_match, method_scope, skill_status, skill_independent, score_band,
  needs_confirmation, min_mistakes.
Chỉ số nghiệp vụ TRỌNG TÂM (chống kết luận oan):
  no_false_gap = ở kịch bản ADVANCED_CORRECT, model KHÔNG gán KNOWLEDGE_GAP.
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict

CHECKS = ["answers_match", "method_scope", "skill_status", "skill_independent",
          "score_band", "needs_confirmation", "min_mistakes"]


def get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict):
            return default
        d = d.get(p)
    return d if d is not None else default


def grade_item(rec) -> dict:
    """Trả dict {check: True/False/None(không áp dụng)} + cờ nghiệp vụ."""
    exp = rec.get("expected") or {}
    res = rec.get("response") or {}
    syl = res.get("syllabus_frame") or {}
    out = {c: None for c in CHECKS}
    flags = {}

    if not rec.get("ok", True):
        return {"checks": out, "flags": {"error": True}}

    # 1. answers_match
    if "answers_match" in exp:
        out["answers_match"] = bool(syl.get("answers_match")) == bool(exp["answers_match"])

    # 2. method_scope ∈ expected (mảng)
    exp_scope = exp.get("method_scope_level")
    got_scope = get(syl, "method_identified", "scope_level")
    if exp_scope and got_scope is not None:
        allow = exp_scope if isinstance(exp_scope, list) else [exp_scope]
        out["method_scope"] = got_scope in allow
        flags["got_scope"] = got_scope

    # 3-4. skill status + independent (khớp theo skill_id)
    got_skills = {s.get("skill_id"): s for s in (syl.get("skills") or [])}
    st_ok, ind_ok, any_gap = [], [], False
    for esk in (exp.get("skills") or []):
        sid = esk.get("skill_id")
        gs = got_skills.get(sid)
        if gs is None:
            st_ok.append(False); ind_ok.append(False); continue
        allow_st = esk.get("status") or []
        allow_st = allow_st if isinstance(allow_st, list) else [allow_st]
        st_ok.append(gs.get("status") in allow_st)
        if "independent" in esk:
            ind_ok.append(bool(gs.get("independent")) == bool(esk["independent"]))
        if gs.get("status") == "KNOWLEDGE_GAP":
            any_gap = True
    if st_ok:
        out["skill_status"] = all(st_ok)
    if ind_ok:
        out["skill_independent"] = all(ind_ok)
    flags["got_status"] = [got_skills.get(e.get("skill_id"), {}).get("status")
                           for e in (exp.get("skills") or [])]

    # 5. score_band
    band = exp.get("score_band")
    qs = res.get("question_score")
    if band and qs is not None:
        out["score_band"] = band[0] <= qs <= band[1]
        flags["got_score"] = qs

    # 6. needs_confirmation
    if "must_flag_confirmation" in exp:
        out["needs_confirmation"] = bool(syl.get("needs_confirmation")) == bool(exp["must_flag_confirmation"])

    # 7. min_mistakes
    if "min_mistakes" in exp:
        out["min_mistakes"] = len(syl.get("mistakes") or []) >= exp["min_mistakes"]

    # Cờ nghiệp vụ: ADVANCED_CORRECT không được gán KNOWLEDGE_GAP (kết luận oan).
    if rec.get("scenario") == "ADVANCED_CORRECT":
        flags["no_false_gap"] = not any_gap
    return {"checks": out, "flags": flags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "graded_*.jsonl")))
    summary = {}
    for fp in files:
        model = os.path.basename(fp)[len("graded_"):-len(".jsonl")]
        # per-check totals, per-scenario item-pass, business flags
        check_tot = defaultdict(lambda: [0, 0])   # check → [pass, applicable]
        scen = defaultdict(lambda: {"items": 0, "item_pass": 0, "errors": 0})
        adv_gap = [0, 0]  # [no_false_gap_count, total_advanced]
        lat = []
        n = 0
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            sc = rec.get("scenario", "?")
            scen[sc]["items"] += 1
            if rec.get("latency_s"):
                lat.append(rec["latency_s"])
            g = grade_item(rec)
            if g["flags"].get("error"):
                scen[sc]["errors"] += 1
                continue
            applic = [c for c in CHECKS if g["checks"][c] is not None]
            passed = [c for c in applic if g["checks"][c]]
            for c in applic:
                check_tot[c][1] += 1
                if g["checks"][c]:
                    check_tot[c][0] += 1
            # item pass = mọi phép áp dụng được đều đúng
            if applic and len(passed) == len(applic):
                scen[sc]["item_pass"] += 1
            if sc == "ADVANCED_CORRECT":
                adv_gap[1] += 1
                if g["flags"].get("no_false_gap"):
                    adv_gap[0] += 1

        def pct(pair):
            return round(100 * pair[0] / pair[1], 1) if pair[1] else None

        summary[model] = {
            "n": n,
            "avg_latency_s": round(sum(lat) / len(lat), 1) if lat else None,
            "checks": {c: {"pass": check_tot[c][0], "of": check_tot[c][1], "pct": pct(check_tot[c])}
                       for c in CHECKS},
            "by_scenario": {s: {**v, "item_pass_pct": round(100 * v["item_pass"] / v["items"], 1) if v["items"] else None}
                            for s, v in sorted(scen.items())},
            "no_false_gap_pct": pct(adv_gap),  # % ca ADVANCED_CORRECT KHÔNG bị oan hổng
            "overall_item_pass_pct": round(
                100 * sum(v["item_pass"] for v in scen.values()) /
                max(1, sum(v["items"] for v in scen.values())), 1),
        }

    out = args.out or os.path.join(args.dir, "summary.json")
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # In bảng gọn
    print(f"{'model':<14}{'n':>4}{'overall%':>9}{'noOan%':>8}{'lat_s':>7}")
    for m, s in summary.items():
        print(f"{m:<14}{s['n']:>4}{str(s['overall_item_pass_pct']):>9}"
              f"{str(s['no_false_gap_pct']):>8}{str(s['avg_latency_s']):>7}")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
