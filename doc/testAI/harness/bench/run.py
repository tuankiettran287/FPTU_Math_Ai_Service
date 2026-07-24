"""Chạy benchmark cho các model thuộc MỘT pha (A hoặc B).

  python -m bench.run --phase A
  python -m bench.run --phase B

Thứ tự chạy cố định (model -> mode -> task theo task_id) và ghi ra JSONL để
có thể dừng/chạy tiếp mà không mất dữ liệu.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from . import config, gpumon, tasks
from .client import chat_stream, wait_ready
from .retrieval import Index, format_context

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_all_tasks() -> tuple[list[tasks.Task], list[dict]]:
    items = tasks.load_test_set(ROOT / "testset")
    v2 = tasks.build_v2(items)
    v1 = tasks.build_v1(items, n_per_subject=10)
    # Dùng lại v3_variants.json nếu đã có: bench.paraphrase gộp thêm biến thể
    # correct_paraphrase vào đó. Sinh lại sẽ xoá mất phần ấy, và tệ hơn là PHA A và
    # PHA B sẽ chạy trên hai bộ test khác nhau -> so sánh vô nghĩa.
    vf = OUT / "v3_variants.json"
    if vf.exists():
        v3_rows = json.loads(vf.read_text(encoding="utf-8"))
        log(f"Dùng lại {len(v3_rows)} biến thể V3 từ v3_variants.json")
    else:
        v3_rows = tasks.build_v3_variants(items, n_per_subject=15)
    v3 = tasks.v3_to_tasks(v3_rows)
    # Bộ khó (nếu có file) — chạy cùng lượt để không phải nạp/tháo model thêm lần nữa.
    v2h = tasks.build_v2_hard(tasks.load_hard_set(ROOT / "testset"))
    if v2h:
        log(f"Có bộ KHÓ: {len(v2h)} câu (V2H)")
    return v2 + v1 + v3 + v2h, v3_rows


def perf_subset(all_tasks: list[tasks.Task]) -> set[str]:
    """Chọn subset đo hiệu năng: chỉ lấy V2, cân bằng môn x độ khó, cố định seed."""
    import random
    rng = random.Random(config.SEED)
    picked: list[str] = []
    v2 = [t for t in all_tasks if t.problem == "V2"]
    for subj in sorted({t.subject for t in v2}):
        per_diff = max(1, config.PERF_SUBSET_PER_SUBJECT // 3)
        for d in ("easy", "medium", "hard"):
            pool = sorted([t.task_id for t in v2 if t.subject == subj and t.difficulty == d])
            rng.shuffle(pool)
            picked += pool[:per_diff]
    return set(picked)


def precompute_contexts(index: Index, all_tasks: list[tasks.Task]) -> dict[str, dict]:
    """Truy hồi TRƯỚC cho mọi task, một lần duy nhất.

    Hai lý do:
      1. Cả 3 model nhận CHÍNH XÁC cùng một ngữ cảnh -> loại biến nhiễu.
      2. Thời gian embed + tìm kiếm không lọt vào số đo TTFT của model.
    """
    log(f"Truy hồi ngữ cảnh cho {len(all_tasks)} task...")
    ctx: dict[str, dict] = {}
    B = 64
    for i in range(0, len(all_tasks), B):
        batch = all_tasks[i:i + B]
        qv = index.embed_query([t.query for t in batch])
        for t, v in zip(batch, qv):
            hits = index.retrieve(t.query, t.subject, qv=v)
            ctx[t.task_id] = {
                "context": format_context(hits),
                "hits": [{"source": h.source, "score": h.score, "ref": h.ref,
                          "subject": h.subject, "text": h.text} for h in hits],
                # Trùng lặp: câu test gần giống hệt một bài trong ngân hàng?
                "max_exemplar_sim": max([h.score for h in hits if h.source == "exemplar"],
                                        default=0.0),
                "max_theory_sim": max([h.score for h in hits if h.source == "theory"],
                                      default=0.0),
                "n_theory": sum(1 for h in hits if h.source == "theory"),
            }
        log(f"  ...{min(i+B, len(all_tasks))}/{len(all_tasks)}")
    return ctx


def run_one(spec, task: tasks.Task, ctx_entry, mode: str, client: httpx.Client) -> dict:
    context = ctx_entry["context"] if mode == "rag" else None
    msgs = tasks.render(task, context)
    r = chat_stream(base_url=spec.base_url, model=spec.hf_id, messages=msgs,
                    temperature=spec.temperature, top_p=spec.top_p,
                    max_tokens=spec.max_tokens, timeout_s=config.REQUEST_TIMEOUT_S,
                    client=client)
    row = {
        "task_id": task.task_id, "problem": task.problem, "model": spec.key,
        "mode": mode, "subject": task.subject, "topic": task.topic,
        "difficulty": task.difficulty, **r.as_row(),
    }
    if mode == "rag":
        row["max_exemplar_sim"] = ctx_entry["max_exemplar_sim"]
        row["max_theory_sim"] = ctx_entry["max_theory_sim"]
        row["n_theory_hits"] = ctx_entry["n_theory"]
        row["cited"] = sorted(tasks.cited_refs(r.text))
    # Bóc kết quả theo từng loại vấn đề + đo Format Compliance.
    if task.problem in ("V2", "V2H"):
        fin = tasks.extract_final(r.text)
        row["final_answer"] = fin
        row["format_ok"] = fin is not None
    elif task.problem == "V3":
        v = tasks.extract_verdict(r.text)
        row["verdict"] = v
        row["label"] = task.meta["label"]
        row["variant"] = task.meta["variant"]
        row["format_ok"] = tasks.VERDICT_MARK in r.text
    elif task.problem == "V1":
        q, a = tasks.extract_generated(r.text)
        row["gen_question"] = q
        row["gen_answer"] = a
        row["format_ok"] = q is not None and a is not None
    return row, r.text


def warmup(spec, client):
    log(f"  làm nóng {spec.key} ({config.WARMUP_REQUESTS} request)...")
    for _ in range(config.WARMUP_REQUESTS):
        chat_stream(base_url=spec.base_url, model=spec.hf_id,
                    messages=[{"role": "user", "content": "Tính 2+2."}],
                    temperature=spec.temperature, top_p=spec.top_p,
                    max_tokens=64, timeout_s=120, client=client)


def main():
    ap = argparse.ArgumentParser()
    # choices PHẢI suy ra từ config.MODELS, không gõ tay: thêm pha mới vào config mà
    # quên sửa dòng này thì run.py chết ngay ở argparse — và nếu nó nằm trong script
    # có `set -e` thì cả pipeline dừng, model đã nạp nằm không đốt tiền GPU.
    # (Đã dính thật: pha "A2" chết ở đây, 27 phút GPU nằm không.)
    ap.add_argument("--phase", required=True,
                    choices=sorted({m.phase for m in config.MODELS}))
    ap.add_argument("--only", default=None, help="chỉ chạy 1 model key")
    ap.add_argument("--limit", type=int, default=None,
                    help="chạy thử: giới hạn N task mỗi vấn đề (V1/V2/V3)")
    ap.add_argument("--suffix", default="", help="hậu tố tên file kết quả (vd '_smoke')")
    ap.add_argument("--problems", default=None,
                    help="chỉ chạy các vấn đề này (vd 'V2H'). Dùng khi bổ sung bộ test "
                         "mới cho model đã chạy xong các vấn đề khác.")
    ap.add_argument("--append", action="store_true",
                    help="GHI THÊM vào file kết quả thay vì ghi đè. BẮT BUỘC khi dùng "
                         "--problems trên model đã có kết quả, nếu không sẽ xoá sạch "
                         "kết quả cũ mà không báo gì.")
    args = ap.parse_args()

    all_tasks_full, v3_rows = build_all_tasks()
    (OUT / "v3_variants.json").write_text(
        json.dumps(v3_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"Tổng task: {len(all_tasks_full)} "
        f"(V1={sum(1 for t in all_tasks_full if t.problem=='V1')}, "
        f"V2={sum(1 for t in all_tasks_full if t.problem=='V2')}, "
        f"V3={sum(1 for t in all_tasks_full if t.problem=='V3')})")

    index = Index(ROOT / "corpus")
    log(f"Chỉ mục: {len(index.chunks_meta)} chunk lý thuyết, {len(index.qb_meta)} bài mẫu")

    # Ngữ cảnh LUÔN truy hồi cho TOÀN BỘ task, kể cả khi --limit: nếu chỉ tính cho
    # phần bị giới hạn thì contexts.json sẽ thiếu, và lần chạy full sau đó dùng lại
    # file thiếu ấy -> KeyError hoặc tệ hơn là chạy sai âm thầm.
    ctx_path = OUT / "contexts.json"
    if ctx_path.exists():
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
        missing = [t.task_id for t in all_tasks_full if t.task_id not in ctx]
        if missing:
            log(f"contexts.json thiếu {len(missing)} task -> truy hồi lại toàn bộ")
            ctx = precompute_contexts(index, all_tasks_full)
            ctx_path.write_text(json.dumps(ctx, ensure_ascii=False), encoding="utf-8")
        else:
            log(f"Dùng lại ngữ cảnh đã truy hồi ({len(ctx)} task)")
    else:
        ctx = precompute_contexts(index, all_tasks_full)
        ctx_path.write_text(json.dumps(ctx, ensure_ascii=False), encoding="utf-8")

    # Subset hiệu năng cũng tính trên bộ ĐẦY ĐỦ -> giữ nguyên dù có --limit hay không.
    subset = perf_subset(all_tasks_full)

    all_tasks = all_tasks_full

    if args.problems:
        want = set(args.problems.split(","))
        all_tasks = [t for t in all_tasks if t.problem in want]
        # Bộ khó KHÔNG đo hiệu năng lại: độ trễ/TTFT/VRAM đã đo trên bộ dễ với CÙNG
        # model, CÙNG tham số sinh. Đo lại chỉ tốn GPU mà không thêm thông tin.
        subset = set()
        log(f"CHỈ chạy {sorted(want)}: {len(all_tasks)} task, "
            f"bỏ pha hiệu năng (đã đo ở lượt trước)")

    if args.limit:
        keep, cnt = [], {}
        for t in all_tasks:
            cnt[t.problem] = cnt.get(t.problem, 0) + 1
            if cnt[t.problem] <= args.limit:
                keep.append(t)
        all_tasks = keep
        subset = {t.task_id for t in all_tasks if t.task_id in subset}
        log(f"CHẠY THỬ: {args.limit} task/vấn đề -> {len(all_tasks)} task, "
            f"subset hiệu năng {len(subset)}")

    log(f"Subset đo hiệu năng (concurrency=1): {len(subset)} câu")

    specs = [m for m in config.MODELS if m.phase == args.phase
             and (args.only is None or m.key == args.only)]
    # Model Load Time do serve_phase.sh ghi lúc thực sự nạp model vào GPU.
    lt_file = OUT / "load_times.json"
    load_times = json.loads(lt_file.read_text(encoding="utf-8")) if lt_file.exists() else {}
    mon = gpumon.Monitor(interval_s=1.0)
    mon.start()
    try:
        for spec in specs:
            log(f"=== {spec.label} — chờ cổng {spec.port} ===")
            load_s = wait_ready(spec.port)
            time.sleep(3)
            # Khoá vào cây tiến trình của ĐÚNG model này -> VRAM đo được là của riêng nó,
            # không lẫn model khác đang ở chung card (pha A có 2 model + bge).
            pids = mon.track_port(spec.port)
            time.sleep(2)
            card_vram, _, idle_power = gpumon.sample_once()
            idle_vram = gpumon.vram_of_pids(pids)
            # Model Load Time thật do serve_phase.sh ghi lại lúc nạp; wait_ready ở đây
            # thường ra ~0s vì model đã sẵn sàng từ trước.
            lt = load_times.get(spec.key)
            log(f"  PID {sorted(pids)} | VRAM riêng model {idle_vram:.0f} MiB "
                f"| toàn card {card_vram:.0f} MiB | load {lt or load_s:.0f}s")
            if not pids:
                log("  ⚠️ không xác định được PID -> VRAM riêng model sẽ bỏ trống")

            with httpx.Client(timeout=httpx.Timeout(config.REQUEST_TIMEOUT_S, connect=30.0),
                              limits=httpx.Limits(max_connections=32)) as client:
                warmup(spec, client)
                for mode in ("pure", "rag"):
                    sfx = args.suffix
                    # "a" khi --append: chạy bổ sung bộ test mới cho model đã có kết quả.
                    # Mở "w" trong trường hợp đó sẽ xoá sạch kết quả cũ — im lặng và
                    # không cứu được (phải chạy lại vài giờ GPU).
                    fmode = "a" if args.append else "w"
                    texts_f = open(OUT / f"texts_{spec.key}_{mode}{sfx}.jsonl", fmode, encoding="utf-8")
                    rows_f = open(OUT / f"rows_{spec.key}_{mode}{sfx}.jsonl", fmode, encoding="utf-8")

                    # ---- Pha 1: HIỆU NĂNG, concurrency = 1 ----
                    perf_tasks = [t for t in all_tasks if t.task_id in subset]
                    if not perf_tasks:
                        # Bỏ HẲN pha này, không chạy với 0 câu: khối dưới sẽ ghi đè
                        # gpu_*_perf.json bằng một cửa sổ rỗng -> xoá mất số đo độ trễ,
                        # VRAM, điện năng thật của lượt chạy trước, im lặng và không cứu được.
                        log(f"  [{spec.key}/{mode}] bỏ pha hiệu năng — giữ nguyên số đo cũ")
                    else:
                        log(f"  [{spec.key}/{mode}] pha hiệu năng: {len(perf_tasks)} câu, conc=1")
                        t0 = mon.mark()
                        for j, t in enumerate(perf_tasks, 1):
                            row, text = run_one(spec, t, ctx[t.task_id], mode, client)
                            row["phase_kind"] = "perf"
                            rows_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            texts_f.write(json.dumps(
                                {"task_id": t.task_id, "mode": mode, "model": spec.key,
                                 "text": text}, ensure_ascii=False) + "\n")
                            if j % 10 == 0:
                                log(f"    perf {j}/{len(perf_tasks)}")
                        w = mon.window(t0, mon.mark(), n_answers=len(perf_tasks))
                        perf_gpu = {"model": spec.key, "mode": mode, "kind": "perf",
                                    "load_time_s": load_times.get(spec.key, load_s),
                                    "idle_vram_mib": idle_vram,
                                    "card_vram_at_idle_mib": card_vram,
                                    "idle_power_w": idle_power,
                                    **{k: getattr(w, k) for k in
                                       ("vram_mean", "vram_peak", "model_vram_mean",
                                        "model_vram_peak", "util_mean", "util_peak",
                                        "power_mean", "power_peak", "duration_s", "energy_wh")}}
                        # Incremental = VRAM phát sinh khi XỬ LÝ request (KV cache của batch),
                        # tính trên VRAM RIÊNG của model, không phải toàn card.
                        perf_gpu["incremental_vram_mib"] = max(w.model_vram_peak - idle_vram, 0.0)
                        (OUT / f"gpu_{spec.key}_{mode}{sfx}_perf.json").write_text(
                            json.dumps(perf_gpu, ensure_ascii=False, indent=1), encoding="utf-8")
                        log(f"    VRAM model {w.model_vram_peak:.0f} MiB "
                            f"(toàn card {w.vram_peak:.0f}) | util TB {w.util_mean:.0f}% "
                            f"| {w.power_mean:.0f}W | {w.energy_wh*1000:.1f} mWh/câu")

                    # ---- Pha 2: CHẤT LƯỢNG, concurrency = 8 ----
                    rest = [t for t in all_tasks if t.task_id not in subset]
                    log(f"  [{spec.key}/{mode}] pha chất lượng: {len(rest)} task, "
                        f"conc={spec.quality_concurrency}")
                    t0 = mon.mark()
                    done = 0
                    with ThreadPoolExecutor(max_workers=spec.quality_concurrency) as ex:
                        futs = {ex.submit(run_one, spec, t, ctx[t.task_id], mode, client): t
                                for t in rest}
                        for fu in as_completed(futs):
                            try:
                                row, text = fu.result()
                            except Exception as e:  # noqa: BLE001
                                t = futs[fu]
                                row = {"task_id": t.task_id, "problem": t.problem,
                                       "model": spec.key, "mode": mode, "ok": False,
                                       "error": f"{type(e).__name__}: {e}"[:200]}
                                text = ""
                            row["phase_kind"] = "quality"
                            rows_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            texts_f.write(json.dumps(
                                {"task_id": row["task_id"], "mode": mode,
                                 "model": spec.key, "text": text}, ensure_ascii=False) + "\n")
                            done += 1
                            if done % 25 == 0:
                                log(f"    quality {done}/{len(rest)}")
                    w = mon.window(t0, mon.mark(), n_answers=len(rest))
                    (OUT / f"gpu_{spec.key}_{mode}{sfx}_quality.json").write_text(
                        json.dumps({"model": spec.key, "mode": mode, "kind": "quality",
                                    **{k: getattr(w, k) for k in
                                       ("vram_mean", "vram_peak", "model_vram_mean",
                                        "model_vram_peak", "util_mean", "util_peak",
                                        "power_mean", "power_peak", "duration_s", "energy_wh")}},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
                    rows_f.close(); texts_f.close()
                    log(f"  [{spec.key}/{mode}] XONG")
    finally:
        mon.stop()
    log("Pha " + args.phase + " hoàn tất.")


if __name__ == "__main__":
    sys.exit(main())
