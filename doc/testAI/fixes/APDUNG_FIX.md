# Áp dụng 7 fix vào AI service — nhật ký 16/07/2026

Nguồn: `../CAN_SUA_AI_SERVICE.md` (rút ra từ benchmark 15/07). File này ghi **đã làm gì**,
**sửa file nào**, và **còn phải làm tay gì** (những việc đụng DB/hạ tầng không thể áp bằng code ở đây).

Repo AI service: `KLTN/FPTU_Math_Ai_Service-main/AI_service/`.

---

## Tổng kết nhanh

| # | Fix | Loại | Trạng thái |
|---|---|---|---|
| 1 | `_has_cjk` guard cho mọi hàm | code | ✅ **ĐÃ ÁP** — gom guard vào `llm.generate_json`, 10/10 hàm dùng |
| 2 | Nâng `TASK_LIMITS.think` | code | ✅ **ĐÃ ÁP** — `config.py` |
| 3 | RAG hại tác vụ đối chiếu | code | ✅ **ĐÃ ÁP** — thêm cờ `EXPLAIN_MISTAKE_USE_RAG` (mặc định bật, chờ đo) |
| 4 | Bóc đáp án chịu `\boxed{}` | code | ✅ **CÓ SẴN** — `utils.extract_final_answer`, competency đã dùng |
| 5 | Xoá tài liệu MAD101 trùng | DB | 📄 **SCRIPT SẴN** — `fix5_dedup_mad101.sql` (chạy tay trên VPS) |
| 6 | Bổ sung lý thuyết MAS291 | dữ liệu | ⏸ **KHÔNG PHẢI CODE** — cần nạp tài liệu, xem ghi chú |
| 7 | 32B chậm cho Chat/Arena | hạ tầng | ⏸ **KHÔNG PHẢI CODE** — hướng 32B-AWQ chưa kiểm chứng |

---

## 1. 🔴 CJK guard cho mọi hàm — ĐÃ ÁP (gom về một chỗ)

**Vấn đề:** trước đây chỉ `grade_essay` có guard rò chữ Hán; 9 hàm còn lại không có →
32B rò CJK vào ~20% câu trả lời SV đọc.

**Cách sửa — gom guard vào đúng MỘT chokepoint thay vì rải ra 10 hàm:**

- `utils.py`: thêm `dict_has_cjk()` — quét **đệ quy** mọi chuỗi trong JSON output
  (feedback, steps[].content, mistakes[].what…). Bản cũ liệt kê tay từng field nên dễ sót.
- `llm.py`: `generate_json()` thêm tham số `cjk_guard_lang`. Khi `="vi"` và output có CJK,
  sinh LẠI 1 lần với chỉ thị nghiêm ngặt (`_CJK_STRICT_SUFFIX`) + nhiệt độ ×0.5. Chỉ nhận
  bản retry nếu nó SẠCH. Thân hàm cũ tách thành `_generate_json_once()`.
- `features.py`: 10 hàm truyền `cjk_guard_lang=language`:
  `generate_questions, grade, grade_essay, explain_mistake, concept_lookup,
  analyze_learning, solve` (cả full lẫn hidden-full của hint), `study_path`.
  `arena_questions` + `generate_from_mistakes` gọi vòng qua `generate_questions` → tự có guard.
  Gợi ý bước (`solve` mode hint) là văn xuôi (`client.chat`, không qua `generate_json`) →
  thêm guard tay ngay tại chỗ.
- `grade_essay`: bỏ khối retry thủ công cũ, dùng guard chung (đúng tinh thần "rút thành
  helper dùng chung"). Xoá luôn `_has_cjk`/`_CJK_RE` cục bộ trong `features.py`.
- `competency.py`: đã có guard riêng từ trước (dùng `utils.has_cjk`) → giữ nguyên.

**Vì sao gom vào `generate_json`:** không hàm nào có thể QUÊN guard được nữa — thêm feature
mới mà gọi `generate_json(..., cjk_guard_lang=language)` là tự có. Đây là chỗ sửa đúng
nguyên nhân gốc (mọi feature đều đi qua đây), không phải vá từng lá.

## 2. 🔴→🟡 Nâng `TASK_LIMITS.think` — ĐÃ ÁP (`config.py`)

| Khoá | Cũ | Mới | Vì sao |
|---|---:|---:|---|
| `solve_full` | 3.500 | **5.500** | đề khó cần tới 4.691 token nghĩ (đo thật); timeout 180→240 |
| `concept_lookup` | 1.500 | **2.500** | dưới p99 tổng thể; timeout 120→140 |
| `arena_generation` | 2.000 | **2.500** | dưới p99 |
| `generate_mcq` | 2.500 | **3.000** | sát p99 ra đề |
| `explain_mistake` | 2.500 | **3.000** | sát p99 |
| `default` | 2.500 | **3.000** | sát p99 |
| `grade_essay` | 4.000 | 4.000 | giữ — p99 chấm bài chỉ 626, rất dư |
| `solve_hint` | 2.500 | 2.500 | giữ — hint ngắn |

> Mức đúng là 🟡 chứ không 🔴: chỉ 1/360 câu khó vượt trần cũ (0,3%), nhưng kiểu hỏng
> nặng (mất hẳn JSON) và nâng trần gần như miễn phí.

## 3. 🟡 RAG hại tác vụ đối chiếu — ĐÃ ÁP (cờ config, chưa tắt)

- `grade` / `grade_essay`: vốn ĐÃ không dùng RAG → đúng, giữ nguyên.
- `explain_mistake`: cùng bản chất "đối chiếu bài sai với đáp án" nên NGHI hại, nhưng
  **chưa đo riêng**. Thêm cờ `EXPLAIN_MISTAKE_USE_RAG` (`config.py`, mặc định `true`).
  Sau khi đo, đặt biến môi trường `EXPLAIN_MISTAKE_USE_RAG=false` để tắt — **không cần sửa code**.
- `concept_lookup`: RAG đúng chỗ (tra khái niệm = model THIẾU kiến thức) → giữ bật.

> Không tắt mù: doc dặn "cần đo riêng trước khi tắt". Cờ này để bật/tắt sau khi có số đo.

## 4. 🟡 Bóc đáp án chịu `\boxed{}`/markdown — ĐÃ CÓ SẴN

`utils.extract_final_answer()` đã port `extract_final()` của harness (ưu tiên `\boxed{}`
đếm ngoặc lồng, rồi `\( \)`, rồi dòng đầu có nội dung). `competency.py` đã dùng ở 4 chỗ
đối chiếu đáp án. `features.py` dùng JSON có cấu trúc (`final_answer` là field riêng) nên
không còn chỗ bóc-từ-text-tự-do nào bị hỏng. **Không cần sửa thêm.**

---

## 5. 🟢 Xoá tài liệu MAD101 trùng — SCRIPT SẴN, chạy tay trên VPS

`fix5_dedup_mad101.sql` — chạy TỪNG BƯỚC trên **DB của AI service** (không phải DB của BE):
bước 1–2 chỉ SELECT để xem trước; bước 3 mới xoá (chunk tự xoá theo CASCADE). Không chạy
tự động ở đây vì DB nằm trên VPS và thao tác xoá là không hoàn tác.

## 6. 🟢 Lý thuyết MAS291 lệch — KHÔNG PHẢI CODE

MAS291 chỉ 198 chunk so với MAD101 1.427. Cần **nạp thêm tài liệu lý thuyết MAS291** qua
luồng upload document sẵn có (`documents.py` + `POST` ingest). Đây là việc bổ sung nội
dung, không phải sửa code. Bổ sung tài liệu là đòn bẩy lớn hơn đổi model cho môn này.

## 7. 🟢 32B chậm cho Chat/Arena — KHÔNG PHẢI CODE (hạ tầng)

P50 34.7s (32B) vs 3.3s (Qwen). 96GB không chở nổi 32B + model nhanh cùng lúc. Hướng
**chưa kiểm chứng**: serve 32B-AWQ (~20GB) để đủ chỗ chạy kèm một model 7B nhanh cho
Chat/Đấu trường. Cần đo trước khi đưa vào kết luận — không sửa gì trong repo lúc này.

---

## Kiểm tra đã làm

- `python -m py_compile AI_service/*.py` → toàn bộ hợp lệ.
- `grep -c "cjk_guard_lang=language" features.py` → 9 (10 điểm gọi; arena/mistake vòng qua
  generate_questions).
- Không còn `_has_cjk`/`_CJK_RE` cục bộ trong `features.py`.

## Còn phải làm (không thuộc phạm vi sửa code)

1. Chạy `fix5_dedup_mad101.sql` trên VPS (sau khi pg_dump sao lưu).
2. Nạp tài liệu lý thuyết MAS291 (#6).
3. Đo `explain_mistake` có/không RAG rồi quyết định cờ `EXPLAIN_MISTAKE_USE_RAG` (#3).
4. (Tuỳ chọn) Thử nghiệm 32B-AWQ cho Chat/Arena (#7).
