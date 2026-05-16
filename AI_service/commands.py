import argparse
import json

from .db import (
    ensure_ai_tables,
    ensure_question_schema,
    fetch_question_document,
    save_class_analytics,
    save_evaluation,
    save_interaction,
    search_question_documents,
    upsert_question_item,
)
from .llm import generate_json_with_retries, load_llm
from .prompts import (
    build_class_analysis_prompt,
    build_classify_question_prompt,
    build_evaluate_answer_prompt,
    build_explain_wrong_prompt,
    build_generation_prompt,
    build_self_assessment_prompt,
    build_solve_uploaded_prompt,
    build_teacher_chat_prompt,
)
from .schemas import (
    build_question_payload,
    load_json_bank,
    normalize_item,
    read_answer_text_from_args,
    read_problem_text_from_args,
    text_for_embedding,
)
from .utils import load_json_file, missing_dependency_error, now_iso, vector_literal, checked_identifier


def _load_connect_and_embedder(args: argparse.Namespace):
    try:
        from psycopg import connect
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise missing_dependency_error(exc) from exc

    embedder = SentenceTransformer(args.embedding_model)
    return connect, embedder


def generate_and_store(args: argparse.Namespace) -> None:
    connect, embedder = _load_connect_and_embedder(args)
    generate_text, tokenizer, model = load_llm(args)
    vector_dim = embedder.get_sentence_embedding_dimension()

    with connect(args.db_url) as conn:
        ensure_question_schema(conn, args.table, vector_dim)
        for index in range(1, args.count + 1):
            prompt = build_generation_prompt(args, index)
            raw_item = generate_json_with_retries(generate_text, tokenizer, model, prompt, args)
            item = normalize_item(raw_item, args)
            embedding = embedder.encode(text_for_embedding(item), normalize_embeddings=True).tolist()
            upsert_question_item(conn, args.table, item, embedding)
            print(f"saved {item['id']} | {item['chapter']} | {item['topic']} | {item['difficulty']['level']}")


def solve_uploaded_and_store(args: argparse.Namespace) -> None:
    question_text = read_problem_text_from_args(args)
    if not question_text:
        raise SystemExit("Pass --question, --question-file, --ocr-text, or --image-path.")

    connect, embedder = _load_connect_and_embedder(args)
    generate_text, tokenizer, model = load_llm(args)
    vector_dim = embedder.get_sentence_embedding_dimension()

    prompt = build_solve_uploaded_prompt(args, question_text)
    raw_item = generate_json_with_retries(generate_text, tokenizer, model, prompt, args)
    item = normalize_item(raw_item, args)
    item["question"]["text"] = item["question"].get("text") or question_text
    item["question"]["latex"] = item["question"].get("latex") or args.latex or None
    item["question"]["image"] = str(args.image_path) if args.image_path else item["question"].get("image")
    item["metadata"]["source"] = "User Uploaded Image" if args.image_path else "User Uploaded Text"
    item["metadata"]["created_by"] = "AI"
    item["metadata"]["verified"] = False
    item["metadata"]["uploaded_at"] = now_iso()

    embedding = embedder.encode(text_for_embedding(item), normalize_embeddings=True).tolist()
    with connect(args.db_url) as conn:
        ensure_question_schema(conn, args.table, vector_dim)
        upsert_question_item(conn, args.table, item, embedding)
    print(f"saved uploaded problem {item['id']} | {item['chapter']} | {item['topic']}")


def import_json(args: argparse.Namespace) -> None:
    connect, embedder = _load_connect_and_embedder(args)
    vector_dim = embedder.get_sentence_embedding_dimension()
    rows = list(load_json_bank(args.input))

    with connect(args.db_url) as conn:
        ensure_question_schema(conn, args.table, vector_dim)
        for row in rows:
            item = normalize_item(row, args)
            embedding = embedder.encode(text_for_embedding(item), normalize_embeddings=True).tolist()
            upsert_question_item(conn, args.table, item, embedding)
            print(f"saved {item['id']}")
    print(f"imported {len(rows)} rows into {args.table}")


def evaluate_answer(args: argparse.Namespace) -> None:
    student_answer = read_answer_text_from_args(args)
    if not student_answer:
        raise SystemExit("Pass --student-answer, --student-answer-file, --answer-ocr-text, or --answer-image-path.")

    connect, embedder = _load_connect_and_embedder(args)
    generate_text, tokenizer, model = load_llm(args)
    vector_dim = embedder.get_sentence_embedding_dimension()

    with connect(args.db_url) as conn:
        ensure_ai_tables(conn, args.table, vector_dim)
        question_payload = build_question_payload(conn, args, fetch_question_document)
        prompt = build_evaluate_answer_prompt(args, question_payload, student_answer)
        feedback = generate_json_with_retries(generate_text, tokenizer, model, prompt, args)
        embedding_text = f"{json.dumps(question_payload, ensure_ascii=False)}\n{student_answer}\n{json.dumps(feedback, ensure_ascii=False)}"
        evaluation_id = save_evaluation(conn, args, feedback, student_answer, embedding_text)
        interaction_id = save_interaction(
            conn,
            args,
            "evaluate_answer",
            {"question": question_payload, "student_answer": student_answer, "rubric": args.rubric},
            feedback,
            embedding_text,
        )
    print(json.dumps({"evaluation_id": evaluation_id, "interaction_id": interaction_id, "result": feedback}, ensure_ascii=False, indent=2))


def explain_wrong_answer(args: argparse.Namespace) -> None:
    student_answer = read_answer_text_from_args(args)
    if not student_answer:
        raise SystemExit("Pass --student-answer, --student-answer-file, --answer-ocr-text, or --answer-image-path.")

    evaluation_payload = load_json_file(args.evaluation_json) if args.evaluation_json else None
    connect, embedder = _load_connect_and_embedder(args)
    generate_text, tokenizer, model = load_llm(args)
    vector_dim = embedder.get_sentence_embedding_dimension()

    with connect(args.db_url) as conn:
        ensure_ai_tables(conn, args.table, vector_dim)
        question_payload = build_question_payload(conn, args, fetch_question_document)
        prompt = build_explain_wrong_prompt(args, question_payload, student_answer, evaluation_payload)
        explanation = generate_json_with_retries(generate_text, tokenizer, model, prompt, args)
        embedding_text = f"{json.dumps(question_payload, ensure_ascii=False)}\n{student_answer}\n{json.dumps(explanation, ensure_ascii=False)}"
        interaction_id = save_interaction(
            conn,
            args,
            "explain_wrong_answer",
            {"question": question_payload, "student_answer": student_answer, "evaluation": evaluation_payload},
            explanation,
            embedding_text,
        )
    print(json.dumps({"interaction_id": interaction_id, "result": explanation}, ensure_ascii=False, indent=2))


def self_assess(args: argparse.Namespace) -> None:
    answers_payload = load_json_file(args.answers_json) if args.answers_json else None
    connect, embedder = _load_connect_and_embedder(args)
    generate_text, tokenizer, model = load_llm(args)
    vector_dim = embedder.get_sentence_embedding_dimension()
    prompt = build_self_assessment_prompt(args, answers_payload)
    result = generate_json_with_retries(generate_text, tokenizer, model, prompt, args)

    with connect(args.db_url) as conn:
        ensure_ai_tables(conn, args.table, vector_dim)
        interaction_id = save_interaction(
            conn,
            args,
            "self_assessment",
            {"course": args.course, "topics": args.topics, "answers": answers_payload},
            result,
            f"{args.course}\n{args.topics}\n{json.dumps(result, ensure_ascii=False)}",
        )
    print(json.dumps({"interaction_id": interaction_id, "result": result}, ensure_ascii=False, indent=2))


def analyze_class(args: argparse.Namespace) -> None:
    if not args.input_json:
        raise SystemExit("Pass --input-json containing class submissions/scores.")

    class_records = load_json_file(args.input_json)
    connect, embedder = _load_connect_and_embedder(args)
    generate_text, tokenizer, model = load_llm(args)
    vector_dim = embedder.get_sentence_embedding_dimension()
    prompt = build_class_analysis_prompt(args, class_records)
    analysis = generate_json_with_retries(generate_text, tokenizer, model, prompt, args)

    with connect(args.db_url) as conn:
        ensure_ai_tables(conn, args.table, vector_dim)
        analytics_id = save_class_analytics(
            conn,
            args,
            class_records,
            analysis,
            f"{args.class_id}\n{args.course}\n{json.dumps(analysis, ensure_ascii=False)}",
        )
        interaction_id = save_interaction(
            conn,
            args,
            "class_analysis",
            {"class_records": class_records},
            analysis,
            f"{args.class_id}\n{json.dumps(analysis, ensure_ascii=False)}",
        )
    print(json.dumps({"analytics_id": analytics_id, "interaction_id": interaction_id, "result": analysis}, ensure_ascii=False, indent=2))


def teacher_chat(args: argparse.Namespace) -> None:
    connect, embedder = _load_connect_and_embedder(args)
    generate_text, tokenizer, model = load_llm(args)
    vector_dim = embedder.get_sentence_embedding_dimension()
    query_embedding = vector_literal(embedder.encode(args.message, normalize_embeddings=True).tolist())
    analytics_payload = load_json_file(args.analytics_json) if args.analytics_json else None

    with connect(args.db_url) as conn:
        ensure_ai_tables(conn, args.table, vector_dim)
        retrieved = search_question_documents(conn, args.table, query_embedding, args.context_limit)
        prompt = build_teacher_chat_prompt(args, retrieved, analytics_payload)
        result = generate_json_with_retries(generate_text, tokenizer, model, prompt, args)
        interaction_id = save_interaction(
            conn,
            args,
            "teacher_chat",
            {"message": args.message, "retrieved_questions": retrieved, "analytics": analytics_payload},
            result,
            f"{args.message}\n{json.dumps(result, ensure_ascii=False)}",
        )
    print(json.dumps({"interaction_id": interaction_id, "result": result}, ensure_ascii=False, indent=2))


def classify_question(args: argparse.Namespace) -> None:
    question_text = read_problem_text_from_args(args)
    if not question_text:
        raise SystemExit("Pass --question, --question-file, --ocr-text, or --image-path.")

    connect, embedder = _load_connect_and_embedder(args)
    generate_text, tokenizer, model = load_llm(args)
    vector_dim = embedder.get_sentence_embedding_dimension()
    prompt = build_classify_question_prompt(args, question_text)
    result = generate_json_with_retries(generate_text, tokenizer, model, prompt, args)

    with connect(args.db_url) as conn:
        ensure_ai_tables(conn, args.table, vector_dim)
        interaction_id = save_interaction(
            conn,
            args,
            "classify_question",
            {"question": question_text},
            result,
            f"{question_text}\n{json.dumps(result, ensure_ascii=False)}",
        )
    print(json.dumps({"interaction_id": interaction_id, "result": result}, ensure_ascii=False, indent=2))


def search(args: argparse.Namespace) -> None:
    try:
        from psycopg import connect, sql
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise missing_dependency_error(exc) from exc

    checked_identifier(args.table, "table name")
    embedder = SentenceTransformer(args.embedding_model)
    query_embedding = vector_literal(embedder.encode(args.query, normalize_embeddings=True).tolist())
    table = sql.Identifier(args.table)

    with connect(args.db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT id, chapter, topic, question->>'text' AS question_text,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM {table}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                ).format(table=table),
                [query_embedding, query_embedding, args.limit],
            )
            for row in cur.fetchall():
                print(
                    json.dumps(
                        {
                            "id": row[0],
                            "chapter": row[1],
                            "topic": row[2],
                            "question": row[3],
                            "similarity": float(row[4]),
                        },
                        ensure_ascii=False,
                    )
                )
