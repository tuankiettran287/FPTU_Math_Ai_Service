import argparse
import json
from typing import Any, Dict, List


def build_generation_prompt(args: argparse.Namespace, index: int) -> str:
    return (
        "Create one new math exercise for the FPTU_MATHAI question bank. "
        "Return exactly one valid JSON object, no markdown, no explanation outside JSON. "
        "The JSON object must have the same top-level fields as Data_Bank.json:\n"
        "id, subject, course, chapter, topic, subtopic, difficulty, question_type, question, "
        "solution, concepts_used, prerequisites, common_mistakes, hints, evaluation, metadata.\n\n"
        "Required nested shape:\n"
        "- difficulty: {level, score, estimated_time_minutes, cognitive_level}\n"
        "- question: {text, latex, image}\n"
        "- solution: {final_answer, steps, alternative_solutions}\n"
        "- each solution step: {step_number, title, content, formula}\n"
        "- evaluation: {answer_verifiable, step_verifiable}\n"
        "- metadata: {source, language, created_by, verified}\n\n"
        "The exercise must be original, mathematically correct, and include a step-by-step answer.\n\n"
        f"Subject: {args.subject}\n"
        f"Course: {args.course}\n"
        f"Chapter: {args.chapter}\n"
        f"Topic: {args.topic}\n"
        f"Subtopic: {args.subtopic or args.topic}\n"
        f"Difficulty: {args.difficulty}\n"
        f"Question type: {args.question_type}\n"
        f"Language: {args.language}\n"
        f"Sample number: {index}"
    )


def build_solve_uploaded_prompt(args: argparse.Namespace, question_text: str) -> str:
    image_note = f"\nImage path: {args.image_path}" if args.image_path else ""
    latex_note = f"\nLaTeX: {args.latex}" if args.latex else ""
    return (
        "A student uploaded a math problem and needs a complete solution. "
        "Solve the problem and return exactly one valid JSON object, no markdown, no explanation outside JSON. "
        "The JSON object must have the same top-level fields as Data_Bank.json:\n"
        "id, subject, course, chapter, topic, subtopic, difficulty, question_type, question, "
        "solution, concepts_used, prerequisites, common_mistakes, hints, evaluation, metadata.\n\n"
        "Required nested shape:\n"
        "- difficulty: {level, score, estimated_time_minutes, cognitive_level}\n"
        "- question: {text, latex, image}\n"
        "- solution: {final_answer, steps, alternative_solutions}\n"
        "- each solution step: {step_number, title, content, formula}\n"
        "- evaluation: {answer_verifiable, step_verifiable}\n"
        "- metadata: {source, language, created_by, verified}\n\n"
        "The question.text must contain the user's uploaded problem text. "
        "The question.image field must contain the image path if an image was uploaded. "
        "The solution must include a final answer and clear step-by-step reasoning.\n\n"
        f"Subject: {args.subject}\n"
        f"Course: {args.course}\n"
        f"Chapter: {args.chapter}\n"
        f"Topic: {args.topic}\n"
        f"Subtopic: {args.subtopic or args.topic}\n"
        f"Difficulty: {args.difficulty}\n"
        f"Question type: {args.question_type}\n"
        f"Language: {args.language}\n\n"
        f"Uploaded problem text:\n{question_text}{latex_note}{image_note}"
    )


def build_evaluate_answer_prompt(
    args: argparse.Namespace,
    question_payload: Dict[str, Any],
    student_answer: str,
) -> str:
    return (
        "Grade the student's math answer. Return exactly one valid JSON object, no markdown. "
        "The first field must be verdict with value DUNG or SAI. Grade step-by-step when possible.\n\n"
        "Required JSON fields: verdict, is_correct, score, max_score, feedback, expected_answer, "
        "mistakes, step_feedback, suggested_fix, confidence.\n\n"
        "Rules:\n"
        "- verdict must be DUNG if the answer is mathematically correct, otherwise SAI.\n"
        "- score must be a number from 0 to max_score.\n"
        "- step_feedback must be an array with each checked step, correctness, and comment.\n"
        "- feedback must be useful for an FPT IT student.\n\n"
        f"Rubric:\n{args.rubric or 'Use mathematical correctness, final answer, and reasoning quality.'}\n\n"
        f"Question JSON:\n{json.dumps(question_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Student answer:\n{student_answer}"
    )


def build_explain_wrong_prompt(
    args: argparse.Namespace,
    question_payload: Dict[str, Any],
    student_answer: str,
    evaluation_payload: Dict[str, Any] | None,
) -> str:
    evaluation_text = json.dumps(evaluation_payload, ensure_ascii=False, indent=2) if evaluation_payload else ""
    return (
        "Explain why the student's math answer is wrong and how to fix it. "
        "Return exactly one valid JSON object, no markdown.\n\n"
        "Required JSON fields: short_reason, detailed_explanation, corrected_solution_steps, "
        "key_concepts_to_review, next_hint, final_answer.\n\n"
        f"Question JSON:\n{json.dumps(question_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Student answer:\n{student_answer}\n\n"
        f"Previous evaluation JSON, if any:\n{evaluation_text}"
    )


def build_self_assessment_prompt(args: argparse.Namespace, answers_payload: Any | None) -> str:
    answers_text = json.dumps(answers_payload, ensure_ascii=False, indent=2) if answers_payload is not None else "null"
    return (
        "Create or grade a self-study math diagnostic assessment for an FPT IT student. "
        "Return exactly one valid JSON object, no markdown.\n\n"
        "Required JSON fields: mode, course, topics, estimated_level, strengths, weaknesses, "
        "recommended_path, diagnostic_questions, scoring_rubric, feedback.\n\n"
        "If answers_json is null, generate diagnostic_questions only. "
        "If answers_json is provided, grade the answers and infer estimated_level.\n\n"
        f"Course: {args.course}\n"
        f"Topics: {args.topics}\n"
        f"Number of questions: {args.num_questions}\n"
        f"Difficulty: {args.difficulty}\n"
        f"Student id: {args.student_id or ''}\n"
        f"answers_json:\n{answers_text}"
    )


def build_class_analysis_prompt(args: argparse.Namespace, class_records: Any) -> str:
    return (
        "Analyze class math performance for a teacher. Return exactly one valid JSON object, no markdown.\n\n"
        "Required JSON fields: class_id, summary, grade_distribution, weak_chapters, weak_topics, "
        "common_mistakes, students_need_support, recommended_actions, suggested_review_questions.\n\n"
        "Use the records to identify which chapters and problem types students got wrong most often. "
        "Keep the output actionable for a teacher.\n\n"
        f"Class id: {args.class_id}\n"
        f"Course: {args.course}\n"
        f"Records JSON:\n{json.dumps(class_records, ensure_ascii=False, indent=2)}"
    )


def build_teacher_chat_prompt(
    args: argparse.Namespace,
    retrieved_questions: List[Dict[str, Any]],
    analytics_payload: Any | None,
) -> str:
    analytics_text = json.dumps(analytics_payload, ensure_ascii=False, indent=2) if analytics_payload is not None else "null"
    return (
        "You are FPTU_MATHAI teacher assistant. Answer the teacher's question using class analytics "
        "and question-bank context. Return exactly one valid JSON object, no markdown.\n\n"
        "Required JSON fields: answer, evidence, suggested_actions, related_questions.\n\n"
        f"Teacher question:\n{args.message}\n\n"
        f"Class id: {args.class_id or ''}\n"
        f"Analytics JSON:\n{analytics_text}\n\n"
        f"Retrieved question-bank context:\n{json.dumps(retrieved_questions, ensure_ascii=False, indent=2)}"
    )


def build_classify_question_prompt(args: argparse.Namespace, question_text: str) -> str:
    return (
        "Classify this math problem for the FPTU_MATHAI question bank. "
        "Return exactly one valid JSON object, no markdown.\n\n"
        "Required JSON fields: subject, course, chapter, topic, subtopic, difficulty, "
        "question_type, concepts_used, prerequisites, estimated_time_minutes, reason.\n\n"
        f"Course hint: {args.course}\n"
        f"Problem:\n{question_text}"
    )
