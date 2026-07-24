"""Đánh giá năng lực Toán học (competency assessment).

LƯU Ý CHẤT LƯỢNG (đo thật trên 32B, smoke test 16/07): luật "phương pháp ngoài
syllabus → ADVANCED_BYPASS" chôn trong prompt chấm dài KHÔNG ăn — model vẫn gán
FPT_CORE cho L'Hôpital. Fix: đối chiếu tên phương pháp với accepted_methods bằng
code; không khớp thì gọi 1 call PHÂN LOẠI CHUYÊN TRÁCH (prompt ngắn + few-shot)
rồi cưỡng chế kết quả bằng code. Không tin LLM tự giữ luật trong prompt dài.

4 nhiệm vụ, đều bám khung năng lực trong bảng competency_framework (dữ liệu mật,
chỉ nạp bằng tools/import_competency_framework.py):

  framework(subject)  — trả cấu trúc khung cho BE (BE cache; FE không gọi AI trực tiếp).
  generate(...)       — sinh câu TỰ LUẬN theo skill mục tiêu + KIỂM ĐỊNH tự động
                        (giải độc lập + sympy đối chiếu, critic 7 tiêu chí, CJK guard).
                        Câu sinh ra KHÔNG lưu vào math_question_bank.
  grade(...)          — chấm 1 câu theo HAI khung độc lập (syllabus CLO/skill và GMC),
                        nhận diện phương pháp thực tế SV dùng, attribution phần AI hỗ trợ.
                        KHÔNG RAG (benchmark: RAG hại tác vụ đối chiếu ở 4/5 model).
  hint(...)           — gợi ý 5 mức leo thang trong lúc làm bài, trả kèm 'revealed'
                        (máy đọc được) để grade() trừ phần AI đã tiết lộ.

Quy tắc phân công với BE: AI chấm và cho điểm TỪNG BÀI; hồ sơ năng lực dài hạn
(mastery/confidence qua nhiều lần đánh giá) do BE tổng hợp bằng code — AI không
tự ghi điểm năng lực cuối cùng vào đâu cả.
"""
from typing import Any

from . import db, rag, verify
from .llm import client
from .utils import extract_final_answer, has_cjk, new_id, now_iso

# ── mức hỗ trợ AI (khớp shared_policy.ai_assistance_levels, bỏ NONE) ──────────
HINT_LEVELS = {
    1: "CONCEPT_HINT",
    2: "METHOD_HINT",
    3: "NEXT_STEP_HINT",
    4: "SOLUTION_OUTLINE",
    5: "FULL_SOLUTION",
}
_LEVEL_CODES = {v: k for k, v in HINT_LEVELS.items()}

METHOD_SCOPE_LEVELS = [
    "FPT_CORE", "FPT_ACCEPTED_ALTERNATIVE", "BEYOND_FPT_VALID",
    "ADVANCED_GENERALIZATION", "INVALID_OR_UNJUSTIFIED",
]
DEMONSTRATION_STATUSES = [
    "DEMONSTRATED", "PARTIALLY_DEMONSTRATED", "NOT_DEMONSTRATED",
    "KNOWLEDGE_GAP", "ADVANCED_BYPASS",
]


def _norm_level(level: Any) -> tuple[int, str]:
    """Chấp nhận cả số (1-5) lẫn code ('METHOD_HINT') → (số, code)."""
    if isinstance(level, str) and level.strip().upper() in _LEVEL_CODES:
        code = level.strip().upper()
        return _LEVEL_CODES[code], code
    try:
        n = int(level)
    except (TypeError, ValueError):
        n = 1
    n = max(1, min(5, n))
    return n, HINT_LEVELS[n]


# ══════════════════════════════════════════════════════════════════════════════
#  Framework — đọc cho BE
# ══════════════════════════════════════════════════════════════════════════════
def framework(*, subject: str) -> dict[str, Any]:
    pack = db.get_framework_subject(subject)
    if pack is None:
        available = db.list_framework_subjects()
        raise ValueError(
            f"Chưa import khung năng lực cho môn '{subject}'. "
            f"Đã có: {available or '(trống — chạy tools/import_competency_framework.py)'}"
        )
    return pack


def _find_skill(subject_payload: dict[str, Any], skill_id: str) -> tuple[dict, dict] | None:
    """Trả (clo, skill) chứa skill_id, hoặc None."""
    for clo in subject_payload.get("clos") or []:
        for sk in clo.get("skills") or []:
            if sk.get("skill_id") == skill_id:
                return clo, sk
    return None


def _find_unit(subject_payload: dict[str, Any], unit_id: str) -> dict[str, Any] | None:
    for u in subject_payload.get("curriculum_units") or []:
        if u.get("unit_id") == unit_id:
            return u
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Sinh đề + kiểm định
# ══════════════════════════════════════════════════════════════════════════════
_GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "question_text": {"type": "string"},
        "latex": {"type": ["string", "null"]},
        "points": {"type": "number"},
        "standard_solution": {"type": "string"},
        "final_answer": {"type": "string"},
        "required_steps": {"type": "array", "items": {"type": "string"}},
        "accepted_methods": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "scope_level": {"type": "string", "enum": METHOD_SCOPE_LEVELS},
                "outline": {"type": "string"},
            },
            "required": ["name", "scope_level"],
        }},
        "grading_notes": {"type": "string"},
        "difficulty": {"type": "string"},
        "topic": {"type": "string"},
    },
    "required": ["question_text", "standard_solution", "final_answer", "accepted_methods"],
}

_SOLVE_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "method_used": {"type": "string"},
        "solvable": {"type": "boolean"},
        "issue": {"type": "string"},
    },
    "required": ["final_answer", "solvable"],
}

_CRITERIA_KEYS = [
    ("statement_correct",     "Đề bài phát biểu ĐÚNG về mặt toán học, không mâu thuẫn nội tại"),
    ("sufficient_data",       "Dữ kiện ĐỦ để giải, không thừa gây nhiễu sai, không thiếu"),
    ("solution_valid",        "Lời giải chuẩn và đáp án ĐÚNG, các bước hợp lệ"),
    ("difficulty_appropriate","Độ khó khớp mức yêu cầu và cognitive level của skill"),
    ("syllabus_fit",          "Nằm TRONG phạm vi syllabus (không đụng excluded topics, không cần kiến thức ngoài chương trình để giải bằng phương pháp chuẩn)"),
    ("measures_target_skill", "Làm được bài này BUỘC PHẢI thể hiện skill/CLO mục tiêu (đo đúng thứ cần đo)"),
    ("methods_enumerated",    "Danh sách accepted_methods đầy đủ các phương pháp hợp lệ chính (kể cả cách ngoài syllabus nếu có), scope_level gán đúng"),
]

_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {"type": "object", "properties": {
            k: {"type": "object", "properties": {
                "passed": {"type": "boolean"},
                "note": {"type": "string"},
                "confidence": {"type": "number"},
            }, "required": ["passed"]} for k, _ in _CRITERIA_KEYS
        }, "required": [k for k, _ in _CRITERIA_KEYS]},
        "overall_pass": {"type": "boolean"},
        "missing_methods": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "scope_level": {"type": "string", "enum": METHOD_SCOPE_LEVELS},
                "outline": {"type": "string"},
            },
            "required": ["name", "scope_level"],
        }},
        "fix_suggestion": {"type": "string"},
    },
    "required": ["criteria", "overall_pass"],
}


def _short_code(subject_payload: dict[str, Any], subject_code: str) -> str:
    """'MAE101' → 'MAE' (khớp cột subject của math_question_bank/document_chunks)."""
    return ((subject_payload.get("subject") or {}).get("short_code")
            or subject_code[:3]).upper()


def _gen_one(subject_code: str, fw: dict[str, Any], target: dict[str, Any],
             difficulty_modifier: float, language: str,
             avoid_texts: list[str]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Sinh 1 câu + kiểm định. Trả (câu đạt | None, report từng bước)."""
    subject_payload = fw["subject"]
    skill_id = str(target.get("skill_id") or "")
    found = _find_skill(subject_payload, skill_id)
    if not found:
        return None, {"stage": "target", "passed": False,
                      "note": f"skill_id '{skill_id}' không tồn tại trong framework"}
    clo, skill = found
    unit = None
    for uid in clo.get("unit_ids") or []:
        unit = _find_unit(subject_payload, uid) or unit

    difficulty = str(target.get("difficulty") or "medium").lower()
    cognitive = str(target.get("cognitive_level") or "") or ", ".join(skill.get("cognitive_levels") or [])
    qgp = subject_payload.get("question_generation_profile") or {}
    subj_meta = subject_payload.get("subject") or {}
    short = _short_code(subject_payload, subject_code)

    # ── nguyên liệu: câu tương tự trong ngân hàng + lý thuyết (RAG — đúng chỗ khi SINH) ──
    seed_query = " ".join(filter(None, [
        subj_meta.get("name_vi") or "", skill.get("name_vi") or "",
        " ".join(skill.get("topics") or []),
    ]))
    similar_qs: list[dict[str, Any]] = []
    theory_ctx = ""
    try:
        from .embeddings import embed_text
        vec = embed_text(seed_query)
        similar_qs = db.search_questions(vec, k=3, filters={"subject": short})
    except Exception:
        pass
    try:
        chunks = rag.retrieve(seed_query, k=3, filters={"subject": short})
        theory_ctx = rag.format_context(chunks)
    except Exception:
        pass
    seeds = []
    for q in similar_qs:
        qt = ((q.get("question") or {}).get("text") or "")[:500]
        if qt:
            seeds.append(qt)

    lang_note = ("Viết TOÀN BỘ bằng Tiếng Việt. TUYỆT ĐỐI KHÔNG dùng chữ Hán."
                 if language == "vi" else "Write entirely in English.")
    mod_note = ""
    if difficulty_modifier > 1.05:
        mod_note = (f"Sinh viên đã đạt điểm cao ở các lần đánh giá trước (hệ số {difficulty_modifier:.2f}) "
                    "→ ra đề Ở CẬN TRÊN của mức độ khó yêu cầu, thêm bước suy luận/biến hoá.")
    elif difficulty_modifier < 0.95:
        mod_note = (f"Sinh viên còn yếu (hệ số {difficulty_modifier:.2f}) → ra đề Ở CẬN DƯỚI "
                    "của mức độ khó yêu cầu, dữ kiện tường minh.")

    system = (
        f"Bạn là chuyên gia ra đề ĐÁNH GIÁ NĂNG LỰC môn {subj_meta.get('name_vi') or subject_code} "
        f"({subject_code}) của Đại học FPT. Sinh ĐÚNG 1 câu TỰ LUẬN yêu cầu trình bày các bước giải "
        "— mục đích là QUAN SÁT cách giải và lập luận của sinh viên, không phải hỏi đáp số.\n"
        f"• Skill cần đo: {skill.get('skill_id')} — {skill.get('name_vi')} "
        f"(topics: {', '.join(skill.get('topics') or [])}; cognitive: {cognitive}).\n"
        f"• CLO: {clo.get('clo_id')} — {clo.get('title_vi')}.\n"
        + (f"• Phạm vi unit: {unit.get('title_vi')} (topics: {', '.join(unit.get('topics') or [])})"
           + (f"; TRÁNH các topic đã cắt: {', '.join(str(x) for x in unit.get('excluded_or_reduced_topics') or [])}"
              if unit.get("excluded_or_reduced_topics") else "") + ".\n" if unit else "")
        + f"• Độ khó: {difficulty}. {mod_note}\n"
        + (f"• Dạng bài ưu tiên: {', '.join(qgp.get('preferred_tasks') or [])}.\n" if qgp.get("preferred_tasks") else "")
        + (f"• TRÁNH: {'; '.join(qgp.get('avoid') or [])}.\n" if qgp.get("avoid") else "")
        + (f"• Lỗi hay gặp của SV (để câu hỏi có thể phát hiện): {'; '.join(skill.get('common_misconceptions') or [])}.\n"
           if skill.get("common_misconceptions") else "")
        + "• Câu hỏi phải MỚI — không sao chép nguyên văn câu mẫu; câu mẫu chỉ là tham khảo phong cách/độ khó.\n"
        + "• 'accepted_methods': liệt kê MỌI phương pháp giải hợp lệ (kể cả phương pháp ngoài syllabus "
          "hoặc tổng quát hơn) kèm scope_level ∈ "
        + "{FPT_CORE, FPT_ACCEPTED_ALTERNATIVE, BEYOND_FPT_VALID, ADVANCED_GENERALIZATION}.\n"
        + "• 'grading_notes': lưu ý cho giám khảo (bẫy thường gặp, tiêu chí cho điểm từng phần).\n"
        + "• 'final_answer': đáp số cuối gọn (nếu bài dạng chứng minh/giải thích thì ghi kết luận cốt lõi).\n"
        + lang_note
    )
    payload: dict[str, Any] = {
        "skill": {"id": skill.get("skill_id"), "name": skill.get("name_vi"),
                  "topics": skill.get("topics") or []},
        "difficulty": difficulty,
        "sample_questions": seeds,
        "theory_context": theory_ctx or "(không có)",
        "language": language,
    }
    if avoid_texts:
        payload["do_not_repeat"] = [t[:300] for t in avoid_texts[-6:]]

    q = client.generate_json("competency_generate", payload, json_schema=_GEN_SCHEMA,
                             system_prompt=system, temperature=0.5)

    # ── deterministic: CJK guard ──────────────────────────────────────────────
    if language == "vi" and has_cjk(q.get("question_text"), q.get("standard_solution"),
                                    q.get("grading_notes")):
        return None, {"stage": "cjk", "passed": False, "note": "output lẫn chữ Hán"}

    # ── giải độc lập (KHÔNG RAG, không nhìn lời giải của generator) ───────────
    solve = client.generate_json(
        "competency_solve_check",
        {"problem": q.get("question_text"), "latex": q.get("latex"), "language": language},
        json_schema=_SOLVE_CHECK_SCHEMA,
        system_prompt=("Giải bài toán từ đầu, độc lập. Nếu đề thiếu dữ kiện/không giải được, "
                       "đặt solvable=false và nêu 'issue'. 'final_answer' ghi đáp số cuối gọn."),
        temperature=0.0,
    )
    answer_check: dict[str, Any] = {"solver_answer": solve.get("final_answer"),
                                    "generator_answer": q.get("final_answer")}
    if not solve.get("solvable", True):
        return None, {"stage": "solve", "passed": False,
                      "note": f"giải độc lập kết luận không giải được: {solve.get('issue')}",
                      "answer_check": answer_check}
    gen_ans = extract_final_answer(q.get("final_answer")) or q.get("final_answer")
    sol_ans = extract_final_answer(solve.get("final_answer")) or solve.get("final_answer")
    eq = verify.answers_equivalent(gen_ans, sol_ans)
    answer_check["equivalent"] = eq
    if eq is False:
        return None, {"stage": "solve", "passed": False,
                      "note": "đáp án generator ≠ đáp án giải độc lập",
                      "answer_check": answer_check}
    # eq is None (không so được bằng sympy — vd bài chứng minh) → giao critic quyết.

    # ── critic 7 tiêu chí (KHÔNG RAG) ─────────────────────────────────────────
    critic_system = (
        "Bạn là hội đồng KIỂM ĐỊNH đề đánh giá năng lực. Chấm NGHIÊM KHẮC từng tiêu chí "
        "— đề lỗi giao cho sinh viên là hỏng cả kết quả đánh giá. Tiêu chí:\n"
        + "\n".join(f"- {k}: {desc}" for k, desc in _CRITERIA_KEYS)
        + "\nNếu accepted_methods THIẾU phương pháp hợp lệ nào, liệt kê vào 'missing_methods'. "
        "Chỉ overall_pass=true khi TẤT CẢ tiêu chí passed."
    )
    critic_payload = {
        "question": q.get("question_text"),
        "standard_solution": q.get("standard_solution"),
        "final_answer": q.get("final_answer"),
        "accepted_methods": q.get("accepted_methods") or [],
        "target_skill": {"id": skill.get("skill_id"), "name": skill.get("name_vi"),
                         "topics": skill.get("topics") or [], "cognitive_levels": cognitive},
        "unit_scope": {"topics": (unit or {}).get("topics") or [],
                       "excluded": (unit or {}).get("excluded_or_reduced_topics") or []},
        "required_difficulty": difficulty,
        "independent_solver_answer_matches": eq,  # None = sympy không so được
        "language": language,
    }
    critic = client.generate_json("competency_critic", critic_payload,
                                  json_schema=_CRITIC_SCHEMA,
                                  system_prompt=critic_system, temperature=0.1)
    criteria = critic.get("criteria") or {}
    failed = [k for k, _ in _CRITERIA_KEYS
              if not (criteria.get(k) or {}).get("passed", False)]
    if failed or not critic.get("overall_pass", False):
        return None, {"stage": "critic", "passed": False,
                      "note": f"trượt tiêu chí: {', '.join(failed) or 'overall'}",
                      "answer_check": answer_check, "criteria": criteria}

    # Critic bổ sung phương pháp thiếu → gộp vào accepted_methods.
    methods = list(q.get("accepted_methods") or [])
    known = {str(m.get("name") or "").strip().lower() for m in methods}
    for m in critic.get("missing_methods") or []:
        if str(m.get("name") or "").strip().lower() not in known:
            methods.append(m)

    question = {
        "id": new_id("CPQ"),
        "skill_id": skill.get("skill_id"),
        "clo_id": clo.get("clo_id"),
        "question_text": q.get("question_text"),
        "latex": q.get("latex"),
        "points": float(q.get("points") or 10.0),
        "standard_solution": q.get("standard_solution"),
        "final_answer": q.get("final_answer"),
        "required_steps": q.get("required_steps") or [],
        "accepted_methods": methods,
        "grading_notes": q.get("grading_notes") or "",
        "difficulty": q.get("difficulty") or difficulty,
        "topic": q.get("topic") or (skill.get("topics") or [""])[0],
        "verification_report": {
            "answer_check": answer_check,
            "criteria": criteria,
            "verified_at": now_iso(),
        },
    }
    return question, {"stage": "done", "passed": True}


def generate(*, subject: str, targets: list[dict[str, Any]],
             mode: str = "CHAPTER_COMPETENCY_ASSESSMENT",
             difficulty_modifier: float = 1.0, language: str = "vi",
             max_attempts_per_question: int = 3) -> dict[str, Any]:
    """Sinh + kiểm định bộ câu hỏi. Mỗi phần tử `targets` → 1 câu.

    KHÔNG lưu câu vào math_question_bank — câu đánh giá năng lực sống trong DB
    của BE (snapshot theo assessment), tách hẳn khỏi ngân hàng câu hỏi.
    """
    if not targets:
        raise ValueError("targets rỗng — BE phải gửi danh sách skill cần đo.")
    fw = framework(subject=subject)

    questions: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    avoid_texts: list[str] = []
    for target in targets:
        got = None
        for attempt in range(1, max_attempts_per_question + 1):
            try:
                got, report = _gen_one(subject, fw, target, difficulty_modifier,
                                       language, avoid_texts)
            except Exception as exc:  # lỗi hạ tầng LLM → thử lại lần nữa
                got, report = None, {"stage": "llm_error", "passed": False, "note": str(exc)}
            if got:
                got["verification_report"]["attempts"] = attempt
                questions.append(got)
                avoid_texts.append(got["question_text"])
                break
            rejects.append({"skill_id": target.get("skill_id"), "attempt": attempt, **report})
        # hết attempts mà không có câu → bỏ target này, BE thấy thiếu qua stats.

    stats = {
        "requested": len(targets),
        "delivered": len(questions),
        "rejected_total": len(rejects),
        "rejected_by_stage": _count_by(rejects, "stage"),
        "rejects": rejects,
    }
    _save("competency_generate", {"subject": subject, "mode": mode,
                                  "targets": targets, "difficulty_modifier": difficulty_modifier},
          {"delivered": len(questions), "stats": {k: v for k, v in stats.items() if k != "rejects"}})
    return {"subject": subject, "mode": mode, "questions": questions, "stats": stats,
            "created_at": now_iso()}


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        k = str(it.get(key) or "?")
        out[k] = out.get(k, 0) + 1
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Chấm — 2 khung độc lập, KHÔNG RAG
# ══════════════════════════════════════════════════════════════════════════════
_SYLLABUS_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "student_final_answer": {"type": "string"},
        "correct_final_answer": {"type": "string"},
        "answers_match": {"type": "boolean"},
        "method_identified": {"type": "object", "properties": {
            "name": {"type": "string"},
            "scope_level": {"type": "string", "enum": METHOD_SCOPE_LEVELS},
            "is_valid": {"type": "boolean"},
            "justification": {"type": "string"},
        }, "required": ["name", "scope_level", "is_valid"]},
        "question_score": {"type": "number"},
        "skills": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "skill_id": {"type": "string"},
                "status": {"type": "string", "enum": DEMONSTRATION_STATUSES},
                "evidence_quote": {"type": "string"},
                "independent": {"type": "boolean"},
                "confidence": {"type": "number"},
                "note": {"type": "string"},
            },
            "required": ["skill_id", "status", "confidence"],
        }},
        "mistakes": {"type": "array", "items": {
            "type": "object",
            "properties": {"step": {"type": "string"}, "what": {"type": "string"}, "why": {"type": "string"}},
            "required": ["what"],
        }},
        "needs_confirmation": {"type": "boolean"},
        "feedback": {"type": "string"},
    },
    "required": ["student_final_answer", "correct_final_answer", "answers_match",
                 "method_identified", "question_score", "skills", "feedback"],
}

def _gmc_grade_schema(dim_ids: list[str]) -> dict[str, Any]:
    """Schema chấm GMC — dimension_id ÉP bằng enum theo đúng danh sách gửi vào
    (guided_json build per-request). Đo thật: để string tự do thì 32B lúc bịa id,
    lúc trả mảng rỗng; enum + minItems bắt nó chấm đủ từng dimension."""
    id_schema: dict[str, Any] = {"type": "string"}
    if dim_ids:
        id_schema = {"type": "string", "enum": dim_ids}
    items: dict[str, Any] = {
        "type": "object",
        "properties": {
            "dimension_id": id_schema,
            "observed": {"type": "boolean"},
            "score": {"type": "number"},
            "justification": {"type": "string"},
            "independent": {"type": "boolean"},
            "confidence": {"type": "number"},
        },
        "required": ["dimension_id", "observed", "score", "confidence"],
    }
    dims_schema: dict[str, Any] = {"type": "array", "items": items}
    if dim_ids:
        dims_schema["minItems"] = len(dim_ids)
        dims_schema["maxItems"] = len(dim_ids)
    return {
        "type": "object",
        "properties": {
            "dimensions": dims_schema,
            "advanced_methods_note": {"type": "string"},
            "feedback": {"type": "string"},
        },
        "required": ["dimensions"],
    }


def _method_matches_accepted(method_name: str, accepted: list[dict[str, Any]]) -> bool:
    """Tên phương pháp SV dùng có trùng (đủ gần) một accepted method không — so token."""
    import re as _re
    mn = set(_re.findall(r"\w+", (method_name or "").lower()))
    if not mn:
        return False
    for m in accepted or []:
        an = set(_re.findall(r"\w+", str(m.get("name") or "").lower()))
        if not an:
            continue
        overlap = len(mn & an) / min(len(mn), len(an))
        if overlap >= 0.6:
            return True
    return False


def _looks_like_no_work(answer: str) -> bool:
    """Bài làm CHỈ có đáp án cuối, KHÔNG có quá trình suy luận?

    Đo thật (benchmark V4, kịch bản RIGHT_ANSWER_NO_WORK): CẢ 5 model thấy đáp số
    đúng là chấm DEMONSTRATED + điểm tối đa, KHÔNG bật needs_confirmation — làm
    sai lệch hồ sơ năng lực (SV chép đáp án vẫn bị/được ghi 'thành thạo'). Không
    quan sát được quá trình thì KHÔNG kết luận năng lực → phải cần bài xác nhận.
    Luật này enforce bằng CODE (không nhét vào prompt dài — cùng bài học ADVANCED_BYPASS).

    Chữ ký 'chỉ có đáp án': 1 dòng nội dung, ≤1 dấu '=', không có từ nối suy luận.
    Bài có trình bày thật (nhiều dòng / nhiều '=' / có 'vì','ta có','suy ra'...) → False.
    """
    import re as _re
    a = (answer or "").strip()
    if not a:
        return False   # bỏ trắng đã xử lý riêng ở đầu grade()
    lines = [ln for ln in a.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return False
    if a.count("=") >= 2:
        return False
    reasoning = _re.search(
        r"(vì|nên|suy ra|ta có|do đó|bởi|xét|đặt|thay|áp dụng|theo|khi đó|=>|⇒|→|"
        r"because|since|therefore|thus|hence|let\b|substitut|by |we have|so that)",
        a, _re.I)
    if reasoning:
        return False
    return True   # 1 dòng, ≤1 '=', không suy luận → chỉ ghi mỗi đáp án


_SCOPE_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "scope_level": {"type": "string", "enum": METHOD_SCOPE_LEVELS},
        "reason": {"type": "string"},
    },
    "required": ["scope_level"],
}


def _classify_method_scope(method: dict[str, Any], accepted: list[dict[str, Any]],
                           scope_topics: str, language: str) -> str | None:
    """Call phân loại CHUYÊN TRÁCH khi phương pháp không khớp accepted_methods.

    Vì sao cần: luật scope nằm trong prompt chấm dài bị 32B bỏ qua (đo thật —
    L'Hôpital vẫn bị gán FPT_CORE). Prompt ngắn, một nhiệm vụ duy nhất + few-shot
    thì model theo được.
    """
    system = (
        "Nhiệm vụ DUY NHẤT: phân loại PHẠM VI của phương pháp sinh viên dùng so với syllabus.\n"
        "- FPT_CORE: trùng bản chất với một phương pháp trong accepted_methods.\n"
        "- FPT_ACCEPTED_ALTERNATIVE: khác accepted_methods nhưng CHỈ dùng kiến thức trong scope_topics.\n"
        "- BEYOND_FPT_VALID: hợp lệ về toán nhưng cần công cụ NGOÀI scope_topics "
        "(kiến thức chương sau / môn khác).\n"
        "- ADVANCED_GENERALIZATION: cách tổng quát/nâng cao hơn hẳn trình độ scope_topics.\n"
        "- INVALID_OR_UNJUSTIFIED: sai hoặc thiếu biện minh.\n"
        "VÍ DỤ 1: scope_topics='limits, limit laws, one-sided limits, continuity'; "
        "accepted=['Biến đổi về giới hạn cơ bản sin(t)/t']; phương pháp='Quy tắc L'Hôpital' "
        "→ BEYOND_FPT_VALID (L'Hôpital cần ĐẠO HÀM — không nằm trong scope_topics).\n"
        "VÍ DỤ 2: scope_topics='factoring, rational expressions, limits'; "
        "accepted=['Rút gọn phân thức rồi lấy giới hạn']; phương pháp='Phân tích nhân tử rồi giản ước' "
        "→ FPT_CORE (cùng bản chất với accepted).\n"
        "VÍ DỤ 3: scope_topics='mathematical induction'; accepted=['Quy nạp toán học']; "
        "phương pháp='Công thức tổng cấp số cộng có sẵn' → FPT_ACCEPTED_ALTERNATIVE nếu công thức "
        "đó thuộc scope, BEYOND_FPT_VALID nếu không."
    )
    payload = {
        "accepted_methods": [str(m.get("name") or "") for m in accepted or []],
        "scope_topics": scope_topics or "(không rõ)",
        "student_method": {"name": method.get("name"), "justification": method.get("justification")},
        "language": language,
    }
    try:
        out = client.generate_json("competency_critic", payload,
                                   json_schema=_SCOPE_CLASSIFY_SCHEMA,
                                   system_prompt=system, temperature=0.0)
        scope = out.get("scope_level")
        return scope if scope in METHOD_SCOPE_LEVELS else None
    except Exception:
        return None


def _hint_context(hint_events: list[dict[str, Any]]) -> tuple[str, int]:
    """Dựng đoạn mô tả hỗ trợ AI cho prompt chấm + mức hỗ trợ cao nhất."""
    if not hint_events:
        return "(sinh viên KHÔNG dùng gợi ý AI — toàn bộ bài làm là độc lập)", 0
    lines = []
    max_level = 0
    for e in hint_events:
        n, code = _norm_level(e.get("level"))
        max_level = max(max_level, n)
        revealed = e.get("revealed") or {}
        parts = []
        for key, label in (("concepts", "khái niệm"), ("methods", "phương pháp"), ("steps", "bước")):
            vals = revealed.get(key) or []
            if vals:
                parts.append(f"{label}: {', '.join(str(v) for v in vals)}")
        draft = str(e.get("draft_at_request") or "").strip()
        lines.append(
            f"- [{code}] AI đã tiết lộ: {('; '.join(parts)) or (str(e.get('hint_text') or '')[:200])}."
            + (f" Bài làm của SV TRƯỚC KHI nhận gợi ý này: «{draft[:400]}»" if draft
               else " (SV chưa viết gì trước khi xin gợi ý này)")
        )
    return "\n".join(lines), max_level


def grade(*, question: dict[str, Any], student_answer: str,
          skill_targets: list[dict[str, Any]],
          gmc_targets: list[dict[str, Any]] | None = None,
          hint_events: list[dict[str, Any]] | None = None,
          language: str = "vi", student_id: str | None = None) -> dict[str, Any]:
    """Chấm 1 câu đánh giá năng lực theo 2 khung ĐỘC LẬP (2 call LLM riêng).

    question: snapshot từ generate() — question_text, standard_solution,
              accepted_methods, grading_notes, points.
    skill_targets: [{skill_id, name, description?}] — skill câu này đo.
    gmc_targets: [{dimension_id, name, indicator?}] — dimension quan sát được.
    hint_events: [{level, hint_text, revealed{concepts,methods,steps}, draft_at_request}].
    """
    events = hint_events or []
    max_points = float(question.get("points") or 10.0)
    hint_ctx, max_hint = _hint_context(events)
    qid = question.get("id")

    # Bỏ trắng → không tốn LLM: mọi skill NOT_DEMONSTRATED (chưa quan sát được,
    # KHÔNG kết luận hổng), mọi dimension không observed.
    if not (student_answer or "").strip():
        return {
            "id": new_id("CPG"), "question_id": qid,
            "question_score": 0.0, "max_points": max_points,
            "syllabus_frame": {
                "method_identified": None,
                "skills": [{"skill_id": s.get("skill_id"), "status": "NOT_DEMONSTRATED",
                            "evidence_quote": "", "independent": True, "confidence": 1.0,
                            "note": "Bỏ trắng — chưa có bằng chứng, không kết luận hổng."}
                           for s in skill_targets],
                "mistakes": [], "needs_confirmation": False,
                "feedback": "Chưa có bài làm." if language == "vi" else "No answer submitted.",
            },
            "general_frame": {"dimensions": [], "feedback": ""},
            "assistance": {"max_level": max_hint, "max_level_code": HINT_LEVELS.get(max_hint, "NONE"),
                           "events_count": len(events), "independence_overridden": False},
            "created_at": now_iso(),
        }

    lang_note = ("Viết TOÀN BỘ bằng Tiếng Việt, TUYỆT ĐỐI KHÔNG dùng chữ Hán."
                 if language == "vi" else "Write entirely in English.")

    # ── Khung 1: syllabus (nhận diện phương pháp thực tế, KHÔNG so cứng lời giải mẫu) ──
    syllabus_system = (
        "Bạn là giám khảo ĐÁNH GIÁ NĂNG LỰC THEO SYLLABUS. Nguyên tắc:\n"
        "1) Bài làm của sinh viên là DỮ LIỆU BẤT BIẾN — không sửa hộ, không tính lại hộ; "
        "trích 'student_final_answer' NGUYÊN VĂN, tự tính 'correct_final_answer', so ra 'answers_match'.\n"
        "2) XÁC ĐỊNH PHƯƠNG PHÁP THỰC TẾ sinh viên dùng ('method_identified') và tự kiểm tra "
        "phương pháp đó có hợp lệ về mặt toán học không — KHÔNG so bài làm với lời giải mẫu; "
        "lời giải mẫu và accepted_methods chỉ để tham khảo. Phương pháp đúng nhưng không có "
        "trong accepted_methods vẫn có thể hợp lệ.\n"
        "3) scope_level — PHẠM VI SYLLABUS được định nghĩa bởi 'accepted_methods_reference' + "
        "'syllabus_scope' (topics của skill đang đo). Gán FPT_CORE/FPT_ACCEPTED_ALTERNATIVE CHỈ KHI "
        "phương pháp SV dùng khớp (hoặc tương đương) một mục trong accepted_methods_reference. "
        "Nếu phương pháp KHÔNG khớp mục nào VÀ dùng công cụ toán vượt phạm vi topics của skill "
        "(vd: dùng đạo hàm/L'Hôpital khi skill chỉ đo biến đổi giới hạn cơ bản; dùng kiến thức "
        "chương sau giải bài chương trước) → BẮT BUỘC BEYOND_FPT_VALID hoặc ADVANCED_GENERALIZATION, "
        "TUYỆT ĐỐI KHÔNG được gán FPT_CORE. INVALID_OR_UNJUSTIFIED = sai hoặc thiếu biện minh.\n"
        "3b) Hệ quả bắt buộc: method scope là BEYOND_FPT_VALID/ADVANCED_GENERALIZATION và bài giải đúng "
        "→ các skill mục tiêu mà kỹ thuật syllabus KHÔNG được thể hiện trực tiếp phải mang status "
        "ADVANCED_BYPASS (không phải DEMONSTRATED) và needs_confirmation=true.\n"
        "4) Với TỪNG skill mục tiêu, gán status: DEMONSTRATED / PARTIALLY_DEMONSTRATED / NOT_DEMONSTRATED "
        "(chưa quan sát được — KHÔNG có nghĩa là không biết) / KNOWLEDGE_GAP (có bằng chứng TRỰC TIẾP "
        "hiểu sai hoặc làm sai kỹ năng) / ADVANCED_BYPASS (dùng phương pháp nâng cao hợp lệ nên kỹ thuật "
        "syllabus không được thể hiện).\n"
        "5) CẤM TUYỆT ĐỐI: sinh viên giải ĐÚNG bằng phương pháp ngoài syllabus/nâng cao → KHÔNG được "
        "gán KNOWLEDGE_GAP; phải gán ADVANCED_BYPASS và đặt needs_confirmation=true.\n"
        "6) 'evidence_quote': trích nguyên văn đoạn bài làm chứng minh cho status.\n"
        "7) HỖ TRỢ AI: phần sinh viên viết SAU khi AI tiết lộ khái niệm/phương pháp/bước nào thì "
        "KHÔNG được tính là bằng chứng độc lập cho đúng thứ đã tiết lộ ('independent'=false cho skill đó). "
        "Đối chiếu draft trước gợi ý với bài nộp cuối để tách phần tự làm/phần sau hỗ trợ.\n"
        "8) 'question_score' (0..max_points): chấm dựa trên PHẦN LÀM ĐỘC LẬP; nếu answers_match=false "
        "thì không được điểm tối đa.\n"
        "9) Mỗi chỗ sai ghi vào 'mistakes' (step trích nguyên văn / what / why).\n"
        "10) 'skills' phải trả ĐÚNG VÀ ĐỦ các skill trong skill_targets, 'skill_id' COPY NGUYÊN VĂN "
        "id được cấp (vd 'MAE-CLO1-S03') — không tự đặt tên khác, không thêm skill lạ. " + lang_note
    )
    syllabus_payload = {
        "question": question.get("question_text"),
        "standard_solution_reference": question.get("standard_solution"),
        "accepted_methods_reference": question.get("accepted_methods") or [],
        "grading_notes": question.get("grading_notes") or "",
        "max_points": max_points,
        "skill_targets": skill_targets,
        "syllabus_scope": [
            {"skill_id": s.get("skill_id"),
             "topics": s.get("topics") or s.get("description") or ""}
            for s in skill_targets
        ],
        "student_answer": student_answer,
        "ai_assistance_log": hint_ctx,
        "language": language,
    }
    syl = client.generate_json("competency_grade", syllabus_payload,
                               json_schema=_SYLLABUS_GRADE_SCHEMA,
                               system_prompt=syllabus_system, temperature=0.2)
    if language == "vi" and has_cjk(syl.get("feedback"),
                                    *[m.get("what") for m in (syl.get("mistakes") or []) if isinstance(m, dict)]):
        retry = client.generate_json("competency_grade", syllabus_payload,
                                     json_schema=_SYLLABUS_GRADE_SCHEMA,
                                     system_prompt=syllabus_system + " CHỈ dùng chữ Latinh và tiếng Việt có dấu.",
                                     temperature=0.1)
        if not has_cjk(retry.get("feedback")):
            syl = retry

    # ── Khung 2: GMC (call RIÊNG để không nhiễm kết luận syllabus) ────────────
    gmc_list = gmc_targets or []
    gmc_system = (
        "Bạn là giám khảo ĐÁNH GIÁ NĂNG LỰC TOÁN HỌC TỔNG QUÁT — KHÔNG quan tâm syllabus hay "
        "chương trình học, chỉ đánh giá năng lực toán thuần tuý thể hiện trong bài làm.\n"
        "1) Trả về ĐỦ TẤT CẢ dimension trong 'dimensions_to_assess' — dimension nào quan sát được "
        "trong bài thì observed=true kèm score; không quan sát được thì observed=false (score=0). "
        "LƯU Ý: một bài giải có trình bày các bước LUÔN quan sát được ít nhất vài dimension "
        "(chọn chiến lược, độ chính xác quy trình, trình bày...) — bài bình thường có 2-5 dimension "
        "observed=true; chỉ toàn observed=false khi bài trống/lạc đề.\n"
        "2) score ∈ [0,1] theo mức thể hiện thật; phương pháp nâng cao/độc đáo hợp lệ được ghi nhận "
        "TÍCH CỰC (ghi vào 'advanced_methods_note').\n"
        "3) HỖ TRỢ AI: phần làm sau khi AI tiết lộ không tính là bằng chứng độc lập (independent=false).\n"
        "4) 'justification' ngắn gọn, dẫn chứng từ bài làm. " + lang_note
    )
    gmc_payload = {
        "question": question.get("question_text"),
        "student_answer": student_answer,
        "dimensions_to_assess": gmc_list,
        "ai_assistance_log": hint_ctx,
        "language": language,
    }
    gmc_dim_ids = [str(g.get("dimension_id") or "").upper() for g in gmc_list
                   if g.get("dimension_id")]
    gmc = client.generate_json("competency_grade", gmc_payload,
                               json_schema=_gmc_grade_schema(gmc_dim_ids),
                               system_prompt=gmc_system, temperature=0.2)

    # ── hậu kiểm bằng code (không tin LLM tự soát) ────────────────────────────
    # sympy đối chiếu đáp án SV với đáp án chuẩn của generate() — độc lập với LLM chấm.
    ref_ans = extract_final_answer(question.get("final_answer")) or question.get("final_answer")
    stu_ans = extract_final_answer(syl.get("student_final_answer")) or syl.get("student_final_answer")
    sympy_check = verify.answers_equivalent(stu_ans, ref_ans)

    score = _clamp(syl.get("question_score"), 0.0, max_points)
    if syl.get("answers_match") is False and score >= max_points and max_points > 0:
        score = round(max_points * 0.5, 2)  # guard "chấm rộng tay" như grade_essay

    # Mức hỗ trợ cao (dàn ý/lời giải đầy đủ) → cưỡng bức independent=false toàn bộ
    # evidence của câu này, bất kể LLM nói gì.
    force_assisted = max_hint >= 4

    # LLM hay TỰ BỊA skill_id (đo thật ở smoke test: trả 'LIMITS_AND_CONTINUITY' thay vì
    # 'MAE-CLO1-S03') → snap kết quả về ĐÚNG danh sách skill_targets bằng code:
    # mỗi target đúng 1 evidence; khớp theo id, không có thì theo thứ tự; target nào
    # model bỏ sót → NOT_DEMONSTRATED confidence thấp (thiếu quan sát ≠ hổng).
    raw_skills = [s for s in (syl.get("skills") or []) if isinstance(s, dict)]
    target_ids = [str(t.get("skill_id") or "") for t in skill_targets]
    by_id = {str(s.get("skill_id") or "").strip().upper(): s for s in raw_skills}
    unmatched = [s for s in raw_skills
                 if str(s.get("skill_id") or "").strip().upper() not in
                 {tid.upper() for tid in target_ids}]
    skills_out = []
    for tid in target_ids:
        s = by_id.get(tid.upper())
        if s is None and unmatched:
            s = unmatched.pop(0)   # model đổi tên id → gán lại theo thứ tự target
        if s is None:
            skills_out.append({
                "skill_id": tid, "status": "NOT_DEMONSTRATED", "evidence_quote": "",
                "independent": True, "confidence": 0.3,
                "note": "AI không đề cập skill này trong kết quả chấm.",
            })
            continue
        independent = bool(s.get("independent", True)) and not force_assisted
        skills_out.append({
            "skill_id": tid,   # LUÔN dùng id chuẩn của target, không tin id model trả
            "status": s.get("status") if s.get("status") in DEMONSTRATION_STATUSES else "NOT_DEMONSTRATED",
            "evidence_quote": s.get("evidence_quote") or "",
            "independent": independent,
            "confidence": _clamp(s.get("confidence"), 0.0, 1.0),
            "note": s.get("note") or "",
        })

    # Hậu kiểm scope — KHÔNG tin phân loại của call chấm (đo thật: L'Hôpital vẫn bị
    # gán FPT_CORE). Quy trình: (1) tên phương pháp khớp accepted_methods bằng code
    # → giữ nguyên; (2) không khớp → call phân loại chuyên trách quyết định scope;
    # (3) scope cuối là BEYOND/ADVANCED + bài đúng → cưỡng chế ADVANCED_BYPASS.
    method = dict(syl.get("method_identified") or {})
    accepted = question.get("accepted_methods") or []
    if (method.get("name") and method.get("is_valid")
            and not _method_matches_accepted(str(method.get("name")), accepted)):
        scope_topics = "; ".join(
            str(s.get("topics") or s.get("description") or "") for s in skill_targets)
        reclassified = _classify_method_scope(method, accepted, scope_topics, language)
        if reclassified:
            method["scope_level"] = reclassified
            method["scope_reclassified_by"] = "dedicated_classifier"
    if (method.get("scope_level") in {"BEYOND_FPT_VALID", "ADVANCED_GENERALIZATION"}
            and method.get("is_valid")):
        for s in skills_out:
            if s["status"] == "DEMONSTRATED":
                s["status"] = "ADVANCED_BYPASS"
                s["note"] = (s["note"] + " " if s["note"] else "") + \
                    "(hệ thống: phương pháp ngoài syllabus → kỹ thuật syllabus chưa quan sát trực tiếp)"

    # Dimension: chỉ giữ id GMC hợp lệ (model bịa id khi thiếu ngữ cảnh — đo thật).
    valid_dims = {str(g.get("dimension_id") or "").upper() for g in (gmc_targets or [])}
    dims_out = []
    for d in (gmc.get("dimensions") or []):
        if not d.get("observed"):
            continue
        did = str(d.get("dimension_id") or "").upper()
        if valid_dims and did not in valid_dims:
            continue
        if not valid_dims and not did.startswith("GMC_"):
            continue
        dims_out.append({
            "dimension_id": did,
            "score": _clamp(d.get("score"), 0.0, 1.0),
            "justification": d.get("justification") or "",
            "independent": bool(d.get("independent", True)) and not force_assisted,
            "confidence": _clamp(d.get("confidence"), 0.0, 1.0),
        })

    # Bài CHỈ có đáp án, không có lời giải → không quan sát được quá trình:
    # không được kết luận thành thạo, phải hạ trần điểm + bật cần-xác-nhận.
    # (đo thật V4: cả 5 model chấm oan DEMONSTRATED+điểm tối đa cho đáp án trần.)
    # CHỈ áp khi đáp án ĐÚNG (đúng định nghĩa RIGHT_ANSWER_NO_WORK: đáp số đúng mà
    # không có lời giải). Bài SAI + no_work (vd lạc đề) KHÔNG đi nhánh này → không bị
    # ép needs_confirmation oan (bài sai chỉ cần làm lại, không cần bài xác nhận).
    # Sửa side-effect đo thật ở V4-after: OFF_TOPIC bị tụt vì needs_confirmation ép sai.
    no_work = _looks_like_no_work(student_answer) and syl.get("answers_match") is True
    if no_work and method.get("scope_level") not in {"BEYOND_FPT_VALID", "ADVANCED_GENERALIZATION"}:
        # phương pháp không kiểm chứng được từ mỗi đáp số
        method["scope_level"] = "INVALID_OR_UNJUSTIFIED"
        method["is_valid"] = False
        for s in skills_out:
            if s["status"] in ("DEMONSTRATED", "PARTIALLY_DEMONSTRATED"):
                s["status"] = "NOT_DEMONSTRATED"
                s["confidence"] = min(_clamp(s.get("confidence"), 0.0, 1.0), 0.4)
                s["note"] = (s["note"] + " " if s["note"] else "") + \
                    "(hệ thống: bài chỉ có đáp án, không có lời giải → chưa quan sát được quá trình)"
        score = min(score, round(max_points * 0.5, 2))   # trần 50% dù đáp án đúng

    # TRẦN ĐIỂM suy ra TỪ trạng thái + đáp án bằng CODE — không tin con số LLM tự cho.
    # (đo thật V4: score_band là phép RỚT NHIỀU NHẤT; ca CALC_SLIP đáp án sai + skill
    # PARTIALLY vẫn bị chấm 8/10 trong khi dải đúng là [4,7]. Điểm phải nhất quán với
    # kết luận định tính — cùng triết lý enforce bằng code như no_work/ADVANCED_BYPASS.)
    elif not any(s["status"] == "ADVANCED_BYPASS" for s in skills_out):   # nâng cao hợp lệ: giữ điểm
        statuses = [s["status"] for s in skills_out]
        ceiling = max_points
        if syl.get("answers_match") is False:
            ceiling = min(ceiling, max_points * 0.6)   # đáp án cuối sai → không thể gần tối đa
        if statuses and all(st in ("NOT_DEMONSTRATED", "KNOWLEDGE_GAP") for st in statuses):
            ceiling = min(ceiling, max_points * 0.4)   # không skill nào thể hiện được
        elif "DEMONSTRATED" not in statuses and any(st == "PARTIALLY_DEMONSTRATED" for st in statuses):
            ceiling = min(ceiling, max_points * 0.7)   # chỉ thể hiện một phần
        score = min(score, round(ceiling, 2))

    advanced_bypass = any(s["status"] == "ADVANCED_BYPASS" for s in skills_out)
    needs_confirmation = (bool(syl.get("needs_confirmation")) or advanced_bypass
                          or max_hint >= 4 or no_work)

    result = {
        "id": new_id("CPG"),
        "question_id": qid,
        "question_score": score if not force_assisted else 0.0,
        "raw_question_score": score,   # điểm chưa trừ hỗ trợ — để audit
        "max_points": max_points,
        "syllabus_frame": {
            "student_final_answer": syl.get("student_final_answer") or "",
            "correct_final_answer": syl.get("correct_final_answer") or "",
            "answers_match": syl.get("answers_match"),
            "sympy_check": sympy_check,   # True/False/None — đối chiếu độc lập bằng code
            "method_identified": {
                "name": method.get("name") or "",
                "scope_level": method.get("scope_level")
                    if method.get("scope_level") in METHOD_SCOPE_LEVELS else "INVALID_OR_UNJUSTIFIED",
                "is_valid": bool(method.get("is_valid")),
                "justification": method.get("justification") or "",
                "scope_reclassified": method.get("scope_reclassified_by") is not None,
            },
            "skills": skills_out,
            "mistakes": [m for m in (syl.get("mistakes") or []) if isinstance(m, dict) and m.get("what")],
            "needs_confirmation": needs_confirmation,
            "feedback": syl.get("feedback") or "",
        },
        "general_frame": {
            "dimensions": dims_out,
            "advanced_methods_note": gmc.get("advanced_methods_note") or "",
            "feedback": gmc.get("feedback") or "",
        },
        "assistance": {
            "max_level": max_hint,
            "max_level_code": HINT_LEVELS.get(max_hint, "NONE"),
            "events_count": len(events),
            "independence_overridden": force_assisted,
        },
        "created_at": now_iso(),
    }
    try:
        db.save_ai_evaluation(student_answer=student_answer, feedback=result,
                              question_id=qid, student_id=student_id,
                              verdict=None, score=result["question_score"], max_score=max_points)
    except Exception:
        pass
    _save("competency_grade", {"question_id": qid, "student_id": student_id,
                               "hint_levels": [(_norm_level(e.get("level")))[1] for e in events]},
          {"score": result["question_score"], "needs_confirmation": needs_confirmation},
          student_id=student_id)
    return result


def _clamp(v: Any, lo: float, hi: float) -> float:
    try:
        return round(max(lo, min(hi, float(v))), 3)
    except (TypeError, ValueError):
        return lo


# ══════════════════════════════════════════════════════════════════════════════
#  Gợi ý 5 mức trong lúc làm bài
# ══════════════════════════════════════════════════════════════════════════════
_HINT_SCHEMA = {
    "type": "object",
    "properties": {
        "hint_text": {"type": "string"},
        "revealed": {"type": "object", "properties": {
            "concepts": {"type": "array", "items": {"type": "string"}},
            "methods": {"type": "array", "items": {"type": "string"}},
            "steps": {"type": "array", "items": {"type": "string"}},
        }},
    },
    "required": ["hint_text", "revealed"],
}

_HINT_RULES = {
    1: ("GỢI Ý KHÁI NIỆM: nhắc lại khái niệm/định nghĩa/định lý liên quan đến bài. "
        "KHÔNG nói dùng phương pháp nào, KHÔNG nói bước làm, KHÔNG lộ đáp án."),
    2: ("GỢI Ý PHƯƠNG PHÁP: nêu TÊN phương pháp/hướng tiếp cận phù hợp và vì sao. "
        "KHÔNG trình bày các bước cụ thể, KHÔNG lộ đáp án."),
    3: ("GỢI Ý BƯỚC TIẾP THEO: dựa trên bài làm dở của sinh viên, gợi mở ĐÚNG 1 bước "
        "kế tiếp. KHÔNG làm thay các bước sau, KHÔNG lộ đáp án cuối."),
    4: ("DÀN Ý LỜI GIẢI: liệt kê khung các bước giải (mỗi bước 1 câu ngắn), "
        "KHÔNG thực hiện tính toán chi tiết, KHÔNG ghi đáp án cuối."),
    5: ("LỜI GIẢI ĐẦY ĐỦ: trình bày lời giải chi tiết từng bước kèm đáp án cuối."),
}


def hint(*, question: dict[str, Any], level: Any = 1, current_draft: str = "",
         subject: str = "", language: str = "vi") -> dict[str, Any]:
    """Gợi ý theo mức leo thang. Trả 'revealed' để grade() attribution.

    Mức 1 (CONCEPT_HINT) được phép RAG lý thuyết (tra khái niệm — đúng chỗ);
    các mức khác KHÔNG RAG (đã có standard_solution trong snapshot).
    """
    n, code = _norm_level(level)
    lang_note = ("Trả lời bằng Tiếng Việt, KHÔNG dùng chữ Hán." if language == "vi"
                 else "Answer in English.")

    theory_ctx = ""
    if n == 1 and subject:
        try:
            chunks = rag.retrieve(str(question.get("question_text") or "")[:500],
                                  k=3, filters={"subject": subject[:3].upper()})
            theory_ctx = rag.format_context(chunks)
        except Exception:
            pass

    system = (
        "Bạn là trợ giảng hỗ trợ sinh viên ĐANG LÀM BÀI ĐÁNH GIÁ NĂNG LỰC. "
        "Có lời giải chuẩn (bí mật) trong tay. Nhiệm vụ: " + _HINT_RULES[n] + "\n"
        "Ngoài ra BẮT BUỘC liệt kê trung thực vào 'revealed' những gì gợi ý này đã tiết lộ "
        "(concepts = khái niệm/định lý đã nhắc tên; methods = phương pháp đã chỉ ra; "
        "steps = các bước đã lộ) — hệ thống dùng danh sách này để phân biệt phần sinh viên "
        "tự làm và phần được hỗ trợ. " + lang_note
    )
    payload = {
        "question": question.get("question_text"),
        "secret_solution": question.get("standard_solution"),
        "secret_final_answer": question.get("final_answer"),
        "student_draft": (current_draft or "").strip() or "(chưa viết gì)",
        "hint_level": code,
        "theory_context": theory_ctx or None,
        "language": language,
    }
    out = client.generate_json("competency_hint", payload, json_schema=_HINT_SCHEMA,
                               system_prompt=system, temperature=0.2)
    if language == "vi" and has_cjk(out.get("hint_text")):
        retry = client.generate_json("competency_hint", payload, json_schema=_HINT_SCHEMA,
                                     system_prompt=system + " CHỈ dùng chữ Latinh và tiếng Việt có dấu.",
                                     temperature=0.1)
        if not has_cjk(retry.get("hint_text")):
            out = retry

    revealed = out.get("revealed") or {}
    result = {
        "id": new_id("CPH"),
        "level": n,
        "level_code": code,
        "hint_text": out.get("hint_text") or "",
        "revealed": {
            "concepts": [str(x) for x in (revealed.get("concepts") or [])],
            "methods": [str(x) for x in (revealed.get("methods") or [])],
            "steps": [str(x) for x in (revealed.get("steps") or [])],
        },
        "created_at": now_iso(),
    }
    _save("competency_hint", {"question_id": question.get("id"), "level": code},
          {"revealed": result["revealed"]})
    return result


# ── helper lưu tương tác ──────────────────────────────────────────────────────
def _save(feature: str, payload: dict[str, Any], output: dict[str, Any], **ids: Any) -> None:
    try:
        db.save_interaction(feature, payload, output, **ids)
    except Exception:
        pass
