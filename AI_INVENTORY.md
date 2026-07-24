# AI Inventory — các tính năng AI hiện có: ĐẦU VÀO → ĐẦU RA

> Service AI: `FPTU_Math_Ai_Service-main` (FastAPI). Base path `/api/v1`. LLM = DeepSeek-R1
> (vLLM), embedding = bge-m3 (1024). Mọi endpoint trả **JSON đầy đủ** để BE dùng ngay.
> BE gọi qua 3 client (`AiServiceClient`, `RagServiceClient`, `ArenaQuestionClient`).

## 1. `POST /generate/questions` — Ra đề (MCQ/tự luận)  ·  màn: Practice, Assignment, Arena, Contest
- **IN**: `subject, count, qtype(mcq|essay), course, chapter, topic, difficulty, language, extra_instructions`
  (đề tự do: chỉ cần `extra_instructions`, để trống subject/chapter).
- **OUT**: `{ questions: [bản ghi câu hỏi đầy đủ: id, question{text,latex,options[]}, solution, evaluation{correct_key}, difficulty, metadata{explanation} ...],
  verification: [{id, distractors(dedup≠đáp án), answer(sympy)}], count }`

## 2. `POST /grade` — Chấm không rubric (2 lớp: sympy + self-consistency)  ·  màn: chấm tự luận Contest/Assignment
- **IN**: `student_answer, question_text|question_id, standard_solution, max_score, language, student_id, class_id, submission_id`
- **OUT**: `GradeResult{ verdict, is_correct, score, max_score, feedback, mistakes[], step_feedback[], suggested_fix, needs_teacher_review, confidence }`

## 3. `POST /explain-mistake` — Giải thích lỗi + lý thuyết (RAG)  ·  màn: sửa câu sai, sổ tay lỗi
- **IN**: `question_text|question_id, student_answer, correct_answer, subject, chapter, concepts[], language`
- **OUT**: `{ error_explanation, correct_concept, steps_to_fix[], sources[] }`

## 4. `POST /concept` — Tra cứu khái niệm (RAG + query rewrite)  ·  màn: Handbook, hỏi lý thuyết
- **IN**: `query, subject, chapter, language`
- **OUT**: `{ answer, sources[{document_id,title,chapter,section}] }`

## 5. `POST /analyze-learning` — Phân tích học tập  ·  màn: **Analytics (SV)**, phân tích lớp (GV)
- **IN**: `student_id, period(week|month), attempts[], errors[], extra_text, language`
- **OUT**: `{ summary, strengths[], weaknesses[], recommendations[], focus_topics[], predicted_trend }`

## 6. `POST /solve` — Giải đầy đủ / gợi ý từng bước (+ ảnh/file)  ·  màn: AI Tutor (Giải bài & Gợi ý)
- **IN**: `problem_text, mode(full|hint), step_index, subject, latex, language, image_base64(→OCR), file_url|file_base64+file_extension(→trích text)`
- **OUT full**: `{ final_answer, steps[{step_number,title,content}], method, answer_check(sympy) }`
- **OUT hint**: `{ mode, step_index, hint, total_steps }` (KHÔNG lộ đáp án)

## 7. `POST /ocr` — OCR ảnh → text/LaTeX  ·  màn: AI Tutor (tải ảnh)
- **IN**: `image_base64, prompt, language` · **OUT**: `{ text, backend, status }`

## 8. `POST /arena/questions` — Ra đề Đấu trường theo ELO  ·  màn: 1v1, Nhiệm vụ, Contest auto
- **IN**: `track(MAE|MAD|MAS), elo, count, chapter, topic, language`
- **OUT**: `{ questions[], elo, difficulty }` (ELO→độ khó)

## 9. `POST /generate/from-mistakes` — Ra đề theo lỗi sai  ·  màn: ôn tập câu sai, SRS
- **IN**: `mistakes[], subject, count, language` · **OUT**: `{ questions[] }`

## 10. `POST /study-path` — Lộ trình học  ·  màn: **Study Path**
- **IN**: `goals/subjects, weak_topics, deadline, level, time_budget, language`
- **OUT**: `{ plan[{day/week, topics[], tasks[]}], milestones[], focus }`

## 11. RAG tài liệu  ·  màn: admin upload (index), AI Tutor tìm tài liệu, Resource Hub
- `POST /documents/ingest` IN `{document_id,file_url,extension,title,subject,chapter,metadata}` → OUT `{document_id, indexed_chunks, status:"indexed"}`
- `POST /documents/search` IN `{query,subject,chapter,limit,allowed_document_ids}` → OUT `{results[{chunk_id,document_id,title,section_title,content,score}]}`
- `POST /documents/ask`   IN `{message,subject,chapter,limit,history,analysis_context}` → OUT `{answer(trích [1][2]), sources[]}`
- `POST /documents/quiz`  IN `{file_url,extension,count,subject,chapter}` → OUT `{questions[], status}`
- `DELETE /documents/{id}` → OUT `{document_id, status:"deleted"}`

## Ai gọi cái gì (BE)
| Client BE | Endpoint AI | Dùng ở |
|---|---|---|
| `ArenaQuestionClient` | /generate/questions (theo mức khó) | 1v1, Nhiệm vụ, Contest auto |
| `AiServiceClient.GenerateExam` | /generate/questions | Practice (ra đề tự do), Assignment |
| `AiServiceClient.SolveProblem` | /solve (full/hint, ảnh/file) | AI Tutor SV |
| `AiServiceClient.ExplainWrongAnswer` | /explain-mistake | sửa câu sai |
| `AiServiceClient.AnalyzeClass` | /analyze-learning | Analytics SV, phân tích lớp GV |
| `RagServiceClient.*` | /documents/{ingest,search,ask,quiz} | admin upload, AI Tutor tìm tài liệu, Resource Hub |
