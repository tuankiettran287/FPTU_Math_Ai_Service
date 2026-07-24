"""9 tính năng AI. Mỗi hàm: dựng prompt (few-shot nơi cần) → gọi LLM 2 giai đoạn
(guided JSON) → verify bằng sympy/code nơi tính được → lưu DB riêng → trả đủ JSON cho BE.
"""
import re
from typing import Any

from . import db, rag, verify
from .config import EXPLAIN_MISTAKE_USE_RAG
from .llm import client
from .utils import has_cjk, new_id, now_iso

# ── track (arena) → môn ───────────────────────────────────────────────────────
TRACK_SUBJECT = {1: "MAE", 2: "MAD", 3: "MAS", "MAE": "MAE", "MAD": "MAD", "MAS": "MAS"}


def _difficulty_from_elo(elo: int) -> str:
    return "easy" if elo < 1000 else "medium" if elo < 1500 else "hard"


# ── schemas (guided_json) ─────────────────────────────────────────────────────
_MCQ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_text": {"type": "string"},
                    "latex": {"type": ["string", "null"]},
                    "options": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}, "text": {"type": "string"}},
                        "required": ["key", "text"],
                    }},
                    "correct_key": {"type": "string"},
                    "explanation": {"type": "string"},
                    "final_answer": {"type": "string"},
                    "topic": {"type": "string"},
                    "difficulty": {"type": "string"},
                },
                "required": ["question_text", "options", "correct_key"],
            },
        }
    },
    "required": ["questions"],
}

_ESSAY_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "question_text": {"type": "string"},
                "latex": {"type": ["string", "null"]},
                "points": {"type": "number"},
                "standard_solution": {"type": "string"},
                "required_steps": {"type": "array", "items": {"type": "string"}},
                "topic": {"type": "string"},
                "difficulty": {"type": "string"},
            },
            "required": ["question_text", "standard_solution"],
        }}
    },
    "required": ["questions"],
}

_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "score": {"type": "number"},
        "is_correct": {"type": "boolean"},
        "required_steps": {"type": "array", "items": {"type": "string"}},
        "step_feedback": {"type": "array", "items": {
            "type": "object",
            "properties": {"step": {"type": "string"}, "ok": {"type": "boolean"}, "comment": {"type": "string"}},
        }},
        "mistakes": {"type": "array", "items": {"type": "string"}},
        "feedback": {"type": "string"},
        "final_answer_student": {"type": "string"},
    },
    "required": ["verdict", "feedback"],
}

# Chấm TỰ LUẬN: khác _GRADE_SCHEMA ở 3 điểm mà nghiệp vụ bắt buộc phải có —
#   - detailed_solution: cách giải chi tiết trả về cho SV
#   - mistakes[]: lỗi sai gắn với TỪNG BƯỚC làm của SV (không phải list string chung chung)
#   - sub_scores[]: điểm từng ý a/b/c
_ESSAY_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        # BẮT BUỘC trích trước, để model phải ĐỌC bài làm thật thay vì tự giải rồi
        # gán kết quả của mình cho sinh viên (nguồn gốc lỗi chấm sai điểm tuyệt đối).
        "student_final_answer": {"type": "string"},   # trích NGUYÊN VĂN đáp số SV viết
        "correct_final_answer": {"type": "string"},   # đáp số đúng do model tự tính
        "answers_match": {"type": "boolean"},         # 2 cái trên có khớp không
        "verdict": {"type": "string"},          # DUNG | MOT_PHAN | SAI
        "score": {"type": "number"},
        "is_correct": {"type": "boolean"},
        "detailed_solution": {"type": "string"},
        "mistakes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "step": {"type": "string"},     # bước nào trong bài làm của SV
                "what": {"type": "string"},     # sai cái gì
                "why": {"type": "string"},      # vì sao sai / đúng ra phải thế nào
            },
            "required": ["what"],
        }},
        "sub_scores": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},      # "a" | "b" | "c"
                "score": {"type": "number"},
                "max_score": {"type": "number"},
                "comment": {"type": "string"},
            },
            "required": ["key", "score"],
        }},
        "feedback": {"type": "string"},
    },
    "required": ["student_final_answer", "correct_final_answer", "answers_match",
                 "verdict", "score", "detailed_solution", "feedback"],
}


# ── 1. Ra đề (MCQ + tự luận) ──────────────────────────────────────────────────
_MCQ_FEWSHOT = (
    "VÍ DỤ (định dạng mong muốn):\n"
    '{"questions":[{"question_text":"Đạo hàm của f(x)=x^2 tại x=3 là?",'
    '"options":[{"key":"A","text":"6"},{"key":"B","text":"9"},{"key":"C","text":"3"},{"key":"D","text":"2x"}],'
    '"correct_key":"A","final_answer":"6","explanation":"f\'(x)=2x, f\'(3)=6.","topic":"Đạo hàm","difficulty":"easy"}]}\n'
    "Yêu cầu: đúng 1 đáp án đúng; các phương án nhiễu KHÁC đáp án đúng và hợp lý."
)


def generate_questions(*, subject: str, count: int = 5, qtype: str = "mcq",
                       course: str = "", chapter: str = "", topic: str = "",
                       difficulty: str = "medium", language: str = "vi",
                       extra_instructions: str = "") -> dict[str, Any]:
    is_mcq = qtype.lower() in {"mcq", "multiple_choice"}
    schema = _MCQ_SCHEMA if is_mcq else _ESSAY_SCHEMA
    task = "generate_mcq" if is_mcq else "generate_essay"
    # Đề tự do: cho phép KHÔNG có môn/chương — khi đó `extra_instructions` (yêu cầu bằng lời
    # của SV/GV) là đặc tả chính. Có môn/chương thì dùng làm phạm vi thu hẹp.
    subject_line = f"môn {subject}" if subject.strip() else "Toán"
    scope_bits = [b for b in (
        f"môn {subject}" if subject.strip() else "",
        f"chương {chapter}" if chapter.strip() else "",
        f"chủ đề {topic}" if topic.strip() else "",
    ) if b]
    scope_line = ("Phạm vi: " + ", ".join(scope_bits) + ".") if scope_bits else \
        "Không giới hạn phạm vi — bám sát đúng yêu cầu bên dưới."
    system = (
        f"Bạn là giáo viên ra đề {subject_line}. Sinh {count} câu "
        + ("TRẮC NGHIỆM (4 phương án, 1 đúng)." if is_mcq else "TỰ LUẬN (kèm lời giải chuẩn + các bước bắt buộc).")
        + f" Độ khó {difficulty}. Ngôn ngữ {language}. "
        + scope_line
        + (f"\n★ YÊU CẦU CỦA NGƯỜI DÙNG (ưu tiên cao nhất, ra đúng dạng này): {extra_instructions}"
           if extra_instructions.strip() else "")
        + (("\n" + _MCQ_FEWSHOT) if is_mcq else "\nMỗi câu nêu rõ 'required_steps' để chấm.")
    )
    payload = {"subject": subject, "course": course, "chapter": chapter, "topic": topic,
               "difficulty": difficulty, "count": count, "type": "mcq" if is_mcq else "essay"}
    result = client.generate_json(task, payload, json_schema=schema, system_prompt=system,
                                  cjk_guard_lang=language)
    raw = result.get("questions") or []

    questions, verification = [], []
    for q in raw:
        qid = new_id("AIQ")
        if is_mcq:
            correct = _correct_text(q)
            dcheck = verify.check_distractors(correct, q.get("options") or [], correct_key=q.get("correct_key"))
            vans = verify.verify_final_answer(q.get("final_answer"), correct) if verify.is_math_like(q.get("final_answer")) else {"verifiable": False}
            verification.append({"id": qid, "distractors": dcheck, "answer": vans})
        record = _to_bank_record(qid, q, subject, course, chapter, topic, difficulty, is_mcq)
        try:
            db.upsert_question(record, None)  # embedding backfill sau
        except Exception:
            pass
        questions.append(record)

    _save("generate_questions", payload, {"count": len(questions)})
    return {"questions": questions, "verification": verification, "count": len(questions)}


def _correct_text(q: dict[str, Any]) -> str | None:
    ck = q.get("correct_key")
    for o in q.get("options") or []:
        if str(o.get("key")).strip().upper() == str(ck).strip().upper():
            return o.get("text")
    return q.get("final_answer")


def _to_bank_record(qid, q, subject, course, chapter, topic, difficulty, is_mcq) -> dict[str, Any]:
    options = [{"key": o.get("key"), "text": o.get("text"), "latex": o.get("latex")}
               for o in (q.get("options") or [])] if is_mcq else []
    return {
        "id": qid, "subject": subject, "course": course, "chapter": chapter,
        "topic": q.get("topic") or topic, "subtopic": "",
        "difficulty": {"level": q.get("difficulty") or difficulty},
        "question_type": "multiple_choice" if is_mcq else "problem_solving",
        "question": {"text": q.get("question_text"), "latex": q.get("latex"), "options": options},
        "solution": {"final_answer": q.get("final_answer") or q.get("standard_solution") or "",
                     "steps": [], "required_steps": q.get("required_steps") or []},
        "concepts_used": [], "prerequisites": [], "common_mistakes": [], "hints": [],
        "evaluation": {"correct_key": q.get("correct_key")} if is_mcq else {"points": q.get("points")},
        "metadata": {"source": "AI", "created_at": now_iso(), "explanation": q.get("explanation")},
    }


# ── 2. Chấm không rubric (2 lớp + self-consistency) ───────────────────────────
def grade(*, student_answer: str, question_text: str = "", question_id: str | None = None,
          standard_solution: Any = None, max_score: float = 10.0, language: str = "vi",
          student_id: str | None = None, class_id: str | None = None,
          submission_id: str | None = None) -> dict[str, Any]:
    q = db.get_question(question_id) if question_id else None
    if q and not question_text:
        question_text = (q.get("question") or {}).get("text", "")
    reference = None
    if q:
        reference = (q.get("solution") or {}).get("final_answer")
    if standard_solution and isinstance(standard_solution, dict):
        reference = standard_solution.get("final_answer") or reference
    elif isinstance(standard_solution, str):
        reference = reference or standard_solution

    # Lớp 1 — sympy (chỉ khi đáp án 'giống toán')
    layer1 = {"applied": False}
    if verify.is_math_like(student_answer) and reference:
        eq = verify.answers_equivalent(student_answer, reference)
        if eq is not None:
            layer1 = {"applied": True, "match": bool(eq)}

    # Lớp 2 — LLM chấm 2 lần độc lập (self-consistency)
    payload = {"question": question_text, "student_answer": student_answer,
               "reference": reference, "standard_solution": standard_solution,
               "max_score": max_score, "language": language}
    system = ("Bạn chấm bài toán KHÔNG theo rubric cứng: bài làm đúng cách nào cũng chấp nhận. "
              "Tự liệt kê các bước BẮT BUỘC để ra kết quả (dựa lời giải chuẩn nếu có), rồi kiểm bài "
              "sinh viên có đủ các bước đó không; chỉ ra bước thiếu/sai, KHÔNG ép đúng thứ tự/cách trình bày.")
    runs = []
    for _ in range(2):
        try:
            runs.append(client.generate_json("grade_submission", payload, json_schema=_GRADE_SCHEMA,
                                              system_prompt=system, temperature=0.35,
                                              cjk_guard_lang=language))
        except Exception as exc:
            runs.append({"verdict": "SAI", "feedback": f"(lỗi chấm: {exc})", "score": 0})

    v0, v1 = _norm_verdict(runs[0]), _norm_verdict(runs[1])
    agree = v0 == v1
    # nếu lớp 1 kết luận chắc → ưu tiên
    if layer1.get("applied"):
        final_verdict = "DUNG" if layer1["match"] else "SAI"
        confidence = 0.95 if agree and (final_verdict == v0) else 0.7
    else:
        final_verdict = v0 if agree else "CAN_REVIEW"
        confidence = 0.85 if agree else 0.4
    needs_review = (not agree) or (confidence < 0.5)

    base = runs[0]
    result = {
        "id": new_id("EVAL"), "question_id": question_id, "verdict": final_verdict,
        "is_correct": final_verdict == "DUNG",
        "score": base.get("score"), "max_score": max_score,
        "feedback": base.get("feedback"), "mistakes": base.get("mistakes") or [],
        "step_feedback": base.get("step_feedback") or [],
        "required_steps": base.get("required_steps") or [],
        "layer1_sympy": layer1, "self_consistency": {"agree": agree, "verdicts": [v0, v1]},
        "confidence": confidence, "needs_teacher_review": needs_review,
        "created_at": now_iso(),
    }
    try:
        db.save_ai_evaluation(student_answer=student_answer, feedback=result, question_id=question_id,
                              student_id=student_id, class_id=class_id, submission_id=submission_id,
                              verdict=final_verdict, score=result.get("score"), max_score=max_score)
    except Exception:
        pass
    return result


# ── 2b. Chấm TỰ LUẬN (có ý a/b/c) — trả cách giải chi tiết + lỗi theo từng bước ──
def grade_essay(*, student_answer: str, question_text: str = "",
                expected_answer: str | None = None,
                sub_questions: list[dict[str, Any]] | None = None,
                max_score: float = 10.0, language: str = "vi",
                question_id: str | None = None, student_id: str | None = None,
                submission_id: str | None = None) -> dict[str, Any]:
    """
    Chấm 1 câu tự luận. Khác `grade`: bám theo CÁC BƯỚC LÀM của SV để chỉ lỗi,
    trả cách giải chi tiết, và chấm điểm từng ý a/b/c nếu câu có chia ý.
    """
    subs = sub_questions or []

    # SV bỏ trắng → 0 điểm, không tốn 1 lượt gọi LLM.
    if not (student_answer or "").strip():
        return {
            "id": new_id("EVAL"), "question_id": question_id,
            "verdict": "SAI", "is_correct": False,
            "score": 0.0, "max_score": max_score,
            "detailed_solution": expected_answer or "",
            "mistakes": [], "sub_scores": [],
            "feedback": "Chưa có bài làm." if language == "vi" else "No answer submitted.",
            "confidence": 1.0, "needs_teacher_review": False,
            "created_at": now_iso(),
        }

    # Lớp 1 — sympy: chỉ dùng để ĐỐI CHIẾU đáp án cuối, không thay được việc chấm từng bước.
    layer1 = {"applied": False}
    if expected_answer and verify.is_math_like(student_answer):
        eq = verify.answers_equivalent(student_answer, expected_answer)
        if eq is not None:
            layer1 = {"applied": True, "match": bool(eq)}

    # Model nền là Qwen → hay chèn chữ Hán vào câu tiếng Việt; phải cấm thẳng.
    lang_note = ("Viết TOÀN BỘ bằng Tiếng Việt. TUYỆT ĐỐI KHÔNG dùng chữ Hán/tiếng Trung "
                 "hay bất kỳ ngôn ngữ nào khác." if language == "vi"
                 else "Answer entirely in English. Do NOT use Chinese characters.")
    system = (
        "Bạn là giám khảo chấm bài toán TỰ LUẬN, chấm theo BƯỚC LÀM của sinh viên, KHÔNG theo rubric cứng: "
        "làm đúng bằng cách nào cũng được điểm.\n"
        "QUY TẮC TỐI QUAN TRỌNG — bài làm của sinh viên là DỮ LIỆU BẤT BIẾN:\n"
        "• TUYỆT ĐỐI KHÔNG được tự sửa, tự làm tròn, hay tự tính lại hộ sinh viên. "
        "Sinh viên viết con số nào thì phải chấm ĐÚNG con số đó, kể cả khi nó sai.\n"
        "• Nếu sinh viên tính sai một phép tính, đó là LỖI và phải bị trừ điểm — "
        "không được coi là 'ý đúng, chỉ nhầm số'.\n"
        "Thứ tự bắt buộc: "
        "(1) 'student_final_answer' = TRÍCH NGUYÊN VĂN đáp số cuối sinh viên viết (copy y hệt, không sửa); "
        "(2) 'correct_final_answer' = tự bạn giải và tính ra đáp số đúng; "
        "(3) 'answers_match' = hai đáp số trên có bằng nhau về GIÁ TRỊ không (so sánh từng con số); "
        "(4) nếu answers_match=false thì verdict KHÔNG được là DUNG và KHÔNG được cho điểm tối đa; "
        "(5) mỗi chỗ sai ghi 1 mục 'mistakes': 'step' (trích nguyên văn bước sai của sinh viên), "
        "'what' (sai cái gì — nêu rõ số sinh viên viết và số đúng), 'why' (vì sao sai); "
        "(6) 'detailed_solution' = cách giải chi tiết ĐÚNG, từng bước; "
        "(7) cho điểm công tâm — đúng một phần thì cho điểm một phần, KHÔNG cho 0 nếu có bước đúng. "
        + ("(8) Câu có nhiều ý: chấm điểm TỪNG Ý vào 'sub_scores', 'key' phải khớp ký hiệu ý (a/b/c); "
           "ý nào có đáp số sai thì ý đó KHÔNG được điểm tối đa. "
           if subs else "")
        + lang_note
    )
    payload: dict[str, Any] = {
        "question": question_text,
        "student_answer": student_answer,
        "expected_answer": expected_answer,
        "max_score": max_score,
        "language": language,
    }
    if subs:
        payload["sub_questions"] = [
            {"key": s.get("key"), "content": s.get("content"),
             "points": s.get("points"), "expected_answer": s.get("expected_answer")}
            for s in subs
        ]

    try:
        # Guard rò chữ Hán (detect + retry) nay nằm trong client.generate_json, quét đệ quy
        # mọi field kể cả mistakes[].what — không cần liệt kê tay từng field như bản cũ.
        raw = client.generate_json("grade_essay", payload, json_schema=_ESSAY_GRADE_SCHEMA,
                                   system_prompt=system, temperature=0.3,
                                   cjk_guard_lang=language)
    except Exception as exc:
        # Chấm hỏng thì KHÔNG được im lặng cho 0 điểm — đánh dấu để người thật xem lại.
        return {
            "id": new_id("EVAL"), "question_id": question_id,
            "verdict": "CAN_REVIEW", "is_correct": False,
            "score": 0.0, "max_score": max_score,
            "detailed_solution": expected_answer or "",
            "mistakes": [], "sub_scores": [],
            "feedback": f"(lỗi chấm: {exc})",
            "confidence": 0.0, "needs_teacher_review": True,
            "created_at": now_iso(),
        }

    # ── Chuẩn hoá điểm: LLM hay trả điểm lệch thang, phải kẹp lại ─────────────
    sub_scores: list[dict[str, Any]] = []
    for s in (raw.get("sub_scores") or []):
        key = _norm_sub_key(s.get("key"))
        declared = next((x for x in subs if _norm_sub_key(x.get("key")) == key), None)
        cap = float(declared.get("points") or 0) if declared else float(s.get("max_score") or 0)
        sub_scores.append({
            "key": key,
            "score": _clamp(s.get("score"), 0.0, cap),
            "max_score": cap,
            "comment": s.get("comment") or "",
        })

    # Model hay BỎ SÓT ý (thường là ý sai). Nếu im lặng cho 0 thì SV mất điểm mà
    # không biết vì sao → điền ý thiếu, ghi rõ là chưa được chấm và đánh dấu soát lại.
    missing_subs = False
    if subs and sub_scores:
        graded_keys = {x["key"] for x in sub_scores}
        for d in subs:
            k = _norm_sub_key(d.get("key"))
            if k not in graded_keys:
                missing_subs = True
                sub_scores.append({
                    "key": k, "score": 0.0, "max_score": float(d.get("points") or 0),
                    "comment": "AI chưa chấm được ý này.",
                })
        sub_scores.sort(key=lambda x: x["key"])

    if sub_scores:
        # Điểm câu = TỔNG điểm các ý (nguồn sự thật), không lấy 'score' rời của LLM.
        score = round(sum(x["score"] for x in sub_scores), 2)
    else:
        score = _clamp(raw.get("score"), 0.0, max_score)

    mistakes = [
        {"step": m.get("step") or "", "what": m.get("what") or "", "why": m.get("why") or ""}
        for m in (raw.get("mistakes") or []) if isinstance(m, dict) and m.get("what")
    ]

    # Chốt chặn chống "chấm rộng tay": model hay tự giải rồi gán kết quả của mình cho SV
    # → tuyên đáp số lệch (answers_match=false) NHƯNG vẫn cho điểm tối đa.
    if raw.get("answers_match") is False and score >= max_score and max_score > 0:
        score = round(max_score * 0.5, 2)   # còn bước đúng thì vẫn có điểm, nhưng không thể trọn vẹn
        sub_scores = []                      # điểm ý đã mâu thuẫn → bỏ, tránh tổng lệch với score

    # Verdict suy từ ĐIỂM, không tin verdict rời của model: đã thấy nó cho 1.5/2
    # mà vẫn tuyên "DUNG" — mâu thuẫn hiển nhiên với chính điểm nó chấm.
    verdict = _verdict_from_score(score, max_score)

    # Các dấu hiệu chấm "không đủ chắc" → gắn cờ để người thật soát lại khi cần:
    #  • sympy nói đáp án cuối khớp mà LLM lại chấm SAI;
    #  • có ý model bỏ sót không chấm;
    #  • trừ điểm nhưng không chỉ ra được lỗi nào (SV không biết sai ở đâu).
    conflict = bool(layer1.get("applied") and layer1["match"] and verdict == "SAI")
    unexplained = verdict != "DUNG" and not mistakes
    shaky = conflict or missing_subs or unexplained

    result = {
        "id": new_id("EVAL"), "question_id": question_id,
        "verdict": verdict,
        "is_correct": verdict == "DUNG",
        "score": score, "max_score": max_score,
        "detailed_solution": raw.get("detailed_solution") or expected_answer or "",
        "mistakes": mistakes,
        "sub_scores": sub_scores,
        "feedback": raw.get("feedback") or "",
        # Để đối chiếu/audit khi nghi ngờ AI chấm sai.
        "student_final_answer": raw.get("student_final_answer") or "",
        "correct_final_answer": raw.get("correct_final_answer") or "",
        "answers_match": raw.get("answers_match"),
        "layer1_sympy": layer1,
        "confidence": 0.5 if shaky else 0.85,
        "needs_teacher_review": shaky,
        "created_at": now_iso(),
    }
    try:
        db.save_ai_evaluation(student_answer=student_answer, feedback=result, question_id=question_id,
                              student_id=student_id, submission_id=submission_id,
                              verdict=verdict, score=score, max_score=max_score)
    except Exception:
        pass
    return result


def _clamp(v: Any, lo: float, hi: float) -> float:
    try:
        return round(max(lo, min(hi, float(v))), 2)
    except (TypeError, ValueError):
        return lo


def _norm_sub_key(raw: Any) -> str:
    """
    Ký hiệu ý về dạng chuẩn 'a'/'b'/'c'.
    Model trả đủ kiểu: 'a', 'a)', 'câu a)', 'Phần b)', 'Part C'…

    KHÔNG lọc theo danh sách tiền tố rồi lấy chữ cái đầu — đã dính bug thật:
    'Phần a)' bị trả về 'p' (chữ đầu của 'phần') vì danh sách thiếu 'phần'
    ⇒ mọi ý normalize thành 'p', trùng key, không khớp ý đã khai → điểm sai.
    Thay vào đó tìm chữ cái a–f ĐỨNG RIÊNG (không nằm trong một từ) — đúng bản chất
    ký hiệu ý, không phụ thuộc việc đoán trước tiền tố nào.

    Ranh giới phải dùng \\w (Unicode), KHÔNG dùng [a-z0-9]: chữ tiếng Việt có dấu
    không nằm trong a-z ASCII nên 'câu a)' sẽ coi 'c' là đứng riêng (vì sau nó là 'â')
    → trả 'c' thay vì 'a'. Đây là bug thật đã bắt được khi test.
    """
    s = str(raw or "").strip().lower()
    m = re.search(r"(?<!\w)([a-f])(?!\w)", s)
    if m:
        return m.group(1)
    m = re.search(r"[a-z]", s)          # dự phòng: key lạ hoàn toàn
    return m.group(0) if m else s.strip()


def _verdict_from_score(score: float, max_score: float) -> str:
    """Verdict LUÔN nhất quán với điểm — điểm là nguồn sự thật duy nhất."""
    if max_score <= 0:
        return "SAI"
    ratio = score / max_score
    if ratio >= 0.99:
        return "DUNG"
    if ratio > 0:
        return "MOT_PHAN"
    return "SAI"


def _norm_verdict(r: dict[str, Any]) -> str:
    v = str(r.get("verdict") or "").strip().upper()
    if v in {"DUNG", "ĐÚNG", "CORRECT", "TRUE"} or r.get("is_correct") is True:
        return "DUNG"
    if v in {"CAN_REVIEW", "REVIEW", "UNSURE"}:
        return "CAN_REVIEW"
    return "SAI"


# ── 3. Giải thích lỗi sai + chỉ điểm lý thuyết (RAG) ──────────────────────────
def explain_mistake(*, question_text: str = "", student_answer: str, subject: str = "",
                    chapter: str = "", concepts: list[str] | None = None,
                    question_id: str | None = None, correct_answer: str = "",
                    language: str = "vi") -> dict[str, Any]:
    # Nếu có question_id, tra câu hỏi trong ngân hàng để có đủ ngữ cảnh.
    if question_id:
        q = db.get_question(question_id)
        if q:
            qq = q.get("question") or {}
            question_text = question_text or qq.get("text", "")
            subject = subject or q.get("subject", "")
            chapter = chapter or q.get("chapter", "")
            if not correct_answer:
                correct_answer = (q.get("solution") or {}).get("final_answer", "")
    # RAG cho explain_mistake: benchmark cho thấy RAG LÀM HẠI các tác vụ ĐỐI CHIẾU bài làm
    # (chấm bài 32B 92.1%→87.0%). explain_mistake cùng bản chất "đối chiếu bài sai với đáp án"
    # nên có thể dính. Chưa đo riêng nên GIỮ RAG bật mặc định, nhưng cho tắt qua config để
    # flip không cần sửa code sau khi đo. Xem CAN_SUA_AI_SERVICE.md mục 3.
    if EXPLAIN_MISTAKE_USE_RAG:
        query = " ".join(x for x in [subject, chapter, question_text] + (concepts or []) if x)
        chunks = rag.retrieve(query, k=4, filters={"subject": subject} if subject else None)
        context = rag.format_context(chunks)
    else:
        chunks = []
        context = ""
    schema = {"type": "object", "properties": {
        "error_explanation": {"type": "string"}, "correct_concept": {"type": "string"},
        "steps_to_fix": {"type": "array", "items": {"type": "string"}},
        "theory_pointers": {"type": "array", "items": {"type": "string"}},
    }, "required": ["error_explanation"]}
    system = ("Giải thích SV sai ở đâu và chỉ ra lý thuyết cần xem. Chỉ dựa vào tài liệu trong CONTEXT "
              "(nếu có); không bịa nguồn. Trích dẫn theo nhãn [1],[2]... của CONTEXT.")
    payload = {"question": question_text, "student_answer": student_answer,
               "correct_answer": correct_answer, "concepts": concepts or [],
               "context": context or "(không có tài liệu)", "language": language}
    out = client.generate_json("explain_mistake", payload, json_schema=schema, system_prompt=system,
                               cjk_guard_lang=language)
    out["sources"] = rag.sources(chunks)
    _save("explain_mistake", payload, out)
    return out


# ── 4. Tra cứu khái niệm (RAG + query rewriting, trả cả file) ─────────────────
def concept_lookup(*, query: str, subject: str = "", chapter: str = "",
                   language: str = "vi") -> dict[str, Any]:
    rq = rag.rewrite_query(query, subject=subject, chapter=chapter)
    chunks = rag.retrieve(rq, k=5, filters={"subject": subject} if subject else None)
    context = rag.format_context(chunks)
    schema = {"type": "object", "properties": {
        "answer": {"type": "string"}, "definition": {"type": "string"},
        "examples": {"type": "array", "items": {"type": "string"}},
        "related_concepts": {"type": "array", "items": {"type": "string"}},
    }, "required": ["answer"]}
    system = ("Trả lời khái niệm toán rõ ràng, ưu tiên dựa CONTEXT tài liệu. Trích dẫn [1],[2]... nếu dùng.")
    payload = {"query": query, "rewritten_query": rq, "context": context or "(không có tài liệu)", "language": language}
    out = client.generate_json("concept_lookup", payload, json_schema=schema, system_prompt=system,
                               cjk_guard_lang=language)
    out["rewritten_query"] = rq
    out["sources"] = rag.sources(chunks)  # cả file tài liệu
    _save("concept_lookup", payload, out)
    return out


# ── 5. Phân tích học tập ──────────────────────────────────────────────────────
def analyze_learning(*, student_id: str | None, period: str = "week",
                     attempts: list[dict[str, Any]] | None = None,
                     errors: list[dict[str, Any]] | None = None,
                     extra_text: str = "", language: str = "vi") -> dict[str, Any]:
    schema = {"type": "object", "properties": {
        "summary": {"type": "string"},
        "mastery_by_topic": {"type": "array", "items": {"type": "object", "properties": {
            "topic": {"type": "string"}, "level": {"type": "string"}, "score": {"type": "number"}}}},
        "error_patterns": {"type": "array", "items": {"type": "string"}},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "trend": {"type": "string"},
        "recommendations": {"type": "array", "items": {"type": "string"}},
    }, "required": ["summary"]}
    system = (f"Phân tích học tập theo {period} (tuần/tháng). Dựa dữ liệu lần làm bài & lỗi sai, đánh giá "
              "mức thành thạo từng chủ đề, mẫu lỗi lặp lại, xu hướng tiến bộ, và đề xuất cụ thể.")
    payload = {"student_id": student_id, "period": period, "attempts": attempts or [],
               "errors": errors or [], "note": extra_text, "language": language}
    out = client.generate_json("learning_analysis", payload, json_schema=schema, system_prompt=system,
                               cjk_guard_lang=language)
    _save("learning_analysis", payload, out, student_id=student_id)
    return out


# ── 6. Giải full vs gợi ý từng bước ───────────────────────────────────────────
def solve(*, problem_text: str = "", mode: str = "full", step_index: int = 1,
          subject: str = "", latex: str | None = None, language: str = "vi",
          image_base64: str = "", file_url: str = "", file_base64: str = "",
          file_extension: str = "") -> dict[str, Any]:
    # Ảnh (đề chụp) → OCR ra text (DeepSeek không nhìn ảnh). File (pdf/docx) → trích text.
    if image_base64:
        from . import ocr as _ocr
        extracted = _ocr.image_to_text(image_base64)
        if extracted:
            problem_text = (problem_text + "\n\n" if problem_text else "") + extracted
    if file_url or file_base64:
        import base64 as _b64
        from . import documents as _docs
        try:
            if file_base64:
                data = file_base64.partition(",")[2] if file_base64.startswith("data:") else file_base64
                file_bytes = _b64.b64decode(data)
            else:
                file_bytes = _docs._download(file_url)
            file_text = _docs.extract_text(file_bytes, file_extension or "pdf")
        except Exception:  # noqa: BLE001
            file_text = ""
        if file_text.strip():
            problem_text = (problem_text + "\n\n" if problem_text else "") + file_text[:6000]
    if not problem_text.strip():
        raise ValueError("Không có nội dung để giải (thiếu đề, hoặc OCR/đọc file không ra chữ).")

    is_en = str(language).lower() == "en"
    lang_rule = " Respond in ENGLISH." if is_en else " Trả lời bằng TIẾNG VIỆT."

    if mode == "hint":
        # Sinh full solution ẩn (không trả về) → lệnh 2 chỉ trích gợi ý bước hiện tại,
        # KHÔNG lộ đáp án cuối.
        full = client.generate_json("solve_full",
                                    {"problem": problem_text, "latex": latex, "language": language},
                                    json_schema=_SOLVE_SCHEMA,
                                    system_prompt="Giải đầy đủ bài toán, liệt kê các bước rõ ràng." + lang_rule,
                                    cjk_guard_lang=language)
        steps = full.get("steps") or []
        hint_sys = ("Give a HINT for the current step based on the hidden solution. "
                    "NEVER reveal the final answer or later steps' results. Only nudge the approach for this step."
                    if is_en else
                    "Bạn đưa GỢI Ý cho bước hiện tại dựa trên lời giải có sẵn. "
                    "TUYỆT ĐỐI KHÔNG tiết lộ đáp án cuối hay kết quả các bước sau. Chỉ gợi mở hướng làm bước này.")
        hint_user = (f"SOLUTION (secret):\n{full}\n\nProblem: {problem_text}\nHint for STEP {step_index} (no answer):"
                     if is_en else
                     f"LỜI GIẢI (bí mật):\n{full}\n\nBài: {problem_text}\nGợi ý cho BƯỚC {step_index} (không lộ đáp án):")

        def _gen_hint(sys_prompt: str, temp: float) -> str:
            return client.chat([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": hint_user},
            ], task="solve_hint", max_new_tokens=300, temperature=temp).strip()

        hint = _gen_hint(hint_sys, 0.2)
        # Gợi ý là văn xuôi (client.chat) nên không qua guard của generate_json — chặn CJK tại đây.
        if not is_en and has_cjk(hint):
            retry = _gen_hint(hint_sys + " CHỈ dùng tiếng Việt có dấu, KHÔNG chữ Hán.", 0.1)
            if not has_cjk(retry):
                hint = retry
        out = {"mode": "hint", "step_index": step_index, "hint": hint,
               "total_steps": len(steps)}
        _save("solve_hint", {"problem": problem_text, "step": step_index}, out)
        return out

    out = client.generate_json("solve_full",
                              {"problem": problem_text, "latex": latex, "language": language},
                              json_schema=_SOLVE_SCHEMA,
                              system_prompt="Giải đầy đủ, kiểm lại đáp án cuối và các bước." + lang_rule,
                              cjk_guard_lang=language)
    # verify đáp án nếu tính được
    fa = out.get("final_answer")
    out["answer_check"] = verify.verify_final_answer(fa, fa) if verify.is_math_like(fa) else {"verifiable": False}
    _save("solve_full", {"problem": problem_text}, out)
    return out


_SOLVE_SCHEMA = {"type": "object", "properties": {
    "final_answer": {"type": "string"},
    "steps": {"type": "array", "items": {"type": "object", "properties": {
        "step_number": {"type": "integer"}, "title": {"type": "string"}, "content": {"type": "string"}}}},
    "method": {"type": "string"},
}, "required": ["final_answer", "steps"]}


# ── 7. Ra đề đấu trường theo ELO (trắc nghiệm) ────────────────────────────────
def arena_questions(*, track: Any, elo: int = 1000, count: int = 5,
                    chapter: str = "", topic: str = "", language: str = "vi") -> dict[str, Any]:
    subject = TRACK_SUBJECT.get(track, "MAE")
    difficulty = _difficulty_from_elo(elo)
    res = generate_questions(subject=subject, count=count, qtype="mcq", chapter=chapter,
                             topic=topic, difficulty=difficulty, language=language,
                             extra_instructions=f"Phù hợp trình độ ELO≈{elo} (đấu trường).")
    res["elo"] = elo
    res["difficulty"] = difficulty
    res["subject"] = subject
    return res


# ── 8. Ra đề theo lỗi sai (kết hợp sổ tay lỗi) ────────────────────────────────
def generate_from_mistakes(*, subject: str, mistakes: list[dict[str, Any]], count: int = 5,
                           qtype: str = "mcq", language: str = "vi") -> dict[str, Any]:
    topics = ", ".join(sorted({str(m.get("topic") or m.get("concept") or "").strip()
                               for m in mistakes if (m.get("topic") or m.get("concept"))}))
    instr = (f"Tập trung vào các điểm yếu/lỗi hay gặp của sinh viên: {topics or 'các lỗi đã ghi'}. "
             "Ra câu rèn đúng vào chỗ sai đó.")
    res = generate_questions(subject=subject, count=count, qtype=qtype, topic=topics[:120],
                             difficulty="medium", language=language, extra_instructions=instr)
    res["targeted_topics"] = topics
    return res


# ── 9. Lộ trình học ───────────────────────────────────────────────────────────
def study_path(*, goal: str = "", level: str = "", subject: str = "",
               weaknesses: list[str] | None = None, deadline: str | None = None,
               weekly_hours: int | None = None, language: str = "vi") -> dict[str, Any]:
    schema = {"type": "object", "properties": {
        "summary": {"type": "string"},
        "focus_areas": {"type": "array", "items": {"type": "string"}},
        "milestones": {"type": "array", "items": {"type": "object", "properties": {
            "order": {"type": "integer"}, "title": {"type": "string"},
            "topics": {"type": "array", "items": {"type": "string"}},
            "activities": {"type": "array", "items": {"type": "string"}},
            "est_hours": {"type": "number"}}}},
    }, "required": ["summary", "milestones"]}
    system = ("Xây LỘ TRÌNH HỌC cá nhân hoá theo mục tiêu/trình độ/điểm yếu. Chia milestone có thứ tự, "
              "mỗi milestone nêu chủ đề, hoạt động (học/luyện/ôn), thời lượng ước tính hợp lý.")
    payload = {"goal": goal, "level": level, "subject": subject, "weaknesses": weaknesses or [],
               "deadline": deadline, "weekly_hours": weekly_hours, "language": language}
    out = client.generate_json("study_path", payload, json_schema=schema, system_prompt=system,
                               cjk_guard_lang=language)
    _save("study_path", payload, out)
    return out


# ── helper lưu tương tác ──────────────────────────────────────────────────────
def _save(feature: str, payload: dict[str, Any], output: dict[str, Any], **ids: Any) -> None:
    try:
        db.save_interaction(feature, payload, output, **ids)
    except Exception:
        pass
