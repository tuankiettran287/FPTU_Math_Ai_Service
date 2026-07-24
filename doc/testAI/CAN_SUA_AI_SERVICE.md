# Những chỗ cần sửa trong AI service — rút ra từ benchmark 15/07/2026

> **Trạng thái: CHƯA SỬA.** File này chỉ ghi nhận. Sửa sau khi chốt báo cáo.
>
> Mọi con số dưới đây là **đo thật** trên 5 model × 2 chế độ × 526 task
> (5.260 lượt sinh), không phải suy đoán. Nguồn: `BAO_CAO_AI.md`,
> `ket_qua_benchmark.xlsx` cùng thư mục.
>
> **Bản sửa 16/07:** đã đối chiếu lại từng con số với dữ liệu thô (`results/graded_*.jsonl`)
> và sửa 4 chỗ sai của bản đầu — xem các ô "Đính chính" trong mục 1, 2, 3, 7 và mục 8.
> Ba con số then chốt đã đổi: rò CJK của 32B **18.2% → 20.2%** (gộp 2 chế độ), RAG hại
> chấm bài **5/5 → 4/5 model**, độ trễ 32B **42s → 34.7s**.

## Tóm tắt việc phải làm

| # | Việc | Mức | File |
|---|---|:--:|---|
| 1 | Áp `_has_cjk` guard cho 9 hàm còn lại | 🔴 | `features.py` |
| 2 | Nâng `TASK_LIMITS.*.think` | 🟡 | `config.py` |
| 3 | Đo riêng `explain_mistake` trước khi tắt RAG | 🟡 | `features.py` |
| 4 | Bóc đáp án chịu được `\boxed{}` + `\( \)` | 🟡 | chỗ nào parse đáp án |
| 5 | Xoá 8 tài liệu MAD101 trùng | 🟢 | DB `document_chunks` |
| 6 | Bổ sung tài liệu lý thuyết MAS291 | 🟢 | DB / kho học liệu |
| 7 | 32B quá chậm cho Chat/Arena — cân nhắc 32B-AWQ | 🟢 | hạ tầng |
| 8 | *(chỉ ghi nhận, không phải lỗi)* Llama yếu toán thật | — | — |

---

## 1. 🔴 `_has_cjk` chỉ bảo vệ 1/10 hàm — sinh viên đang thấy chữ Hán

**Đo được:** DeepSeek-R1-Distill-32B (model production hiện tại) chèn chữ Trung/Nhật/Hàn
vào **20.2%** câu trả lời — tức **cứ ~5 lời giải thì 1 cái có chữ Hán**. Model khác họ
(Llama-3.1-8B) rò **0.2%**, nên đây là đặc tính riêng của dòng R1-Distill (chưng cất từ
dữ liệu nặng tiếng Trung), không phải hiện tượng chung của LLM.

| Model | Rò vào đáp án (gộp 2 chế độ) |
|---|---:|
| Llama-3.1-8B (ngoài họ Qwen) | **0.2%** |
| Qwen2.5-7B | 2.2% |
| R1-Distill-14B | 16.4% |
| R1-Distill-7B | 17.5% |
| **R1-Distill-32B (đang chạy production)** | **20.2%** |

> Con số là **gộp cả chế độ thuần lẫn RAG** (rò CJK không phụ thuộc RAG — nó là đặc
> tính của model). Tách ra: 32B rò 18.2% ở chế độ thuần và 22.2% ở chế độ RAG.

Ví dụ thật lấy từ log:
```
"Tuy nhiên, sinh viên đã tính ra 19π cm²/s, có thể他们是 đã tính sai số"
                                              ^^^^^^
```

**Hiện trạng** (`AI_service/features.py`):

| Hàm | Có `_has_cjk` guard? |
|---|---|
| `grade_essay` | ✅ có (dòng ~361, kèm retry) |
| `solve` | ❌ **không** |
| `generate_questions` | ❌ **không** |
| `arena_questions` | ❌ **không** |
| `grade` | ❌ **không** |
| `explain_mistake` | ❌ **không** |
| `concept_lookup` | ❌ **không** |
| `analyze_learning` | ❌ **không** |
| `generate_from_mistakes` | ❌ **không** |
| `study_path` | ❌ **không** |

**Cần làm:** rút cơ chế guard+retry của `grade_essay` (dòng 361-366) thành helper dùng
chung, áp cho mọi hàm có output hiển thị cho người dùng. Ưu tiên cao nhất: `solve`,
`generate_questions`, `arena_questions` — đây là 3 hàm sinh viên đọc trực tiếp.

---

## 2. 🟡 `TASK_LIMITS.think` thấp hơn ĐỈNH nhu cầu của 32B (hiếm, nhưng hỏng thì hỏng hẳn)

**Đo được** — phân bố token trong thẻ `<think>` của 32B (đếm lại trực tiếp từ
`results/graded_r1_32b_*.jsonl`):

| Nhiệm vụ | n | TB | p50 | p95 | **p99** | **max** |
|---|---:|---:|---:|---:|---:|---:|
| Ra đề (V1) | 54 | 847 | 671 | 1.788 | 2.072 | 2.327 |
| Giải toán — đề dễ | 360 | 421 | 365 | 867 | 1.129 | 1.648 |
| **Giải toán — đề khó** | 360 | 658 | 510 | 1.489 | **2.504** | **4.691** |
| Chấm bài (V3) | 278 | 279 | 258 | 488 | 584 | 1.535 |
| **Tổng** | 1.052 | 486 | 361 | 1.213 | **2.030** | **4.691** |

**Tần suất thật sự bị cắt:** trong 360 câu đề khó, **đúng 1 câu (0,3%)** vượt ngưỡng
3.500 hiện tại. Tức đây **không phải** lỗi xảy ra thường xuyên.

**Nhưng vẫn nên sửa, vì KIỂU hỏng rất xấu:** R1 sinh `<think>` TRƯỚC rồi mới in JSON.
Hết ngân sách `think` giữa chừng thì **JSON không bao giờ được in ra** — request hỏng
hẳn (không phải suy giảm dần), mà log không báo lỗi rõ ràng. Nâng trần gần như miễn phí:
ngân sách thừa không tốn gì nếu model không dùng tới. (Đã dính đúng lỗi này khi làm
benchmark: giám khảo đặt `max_tokens=1024` → mất nhãn hàng loạt; phải nâng lên 3072.)

> Ghi chú trung thực: bản ghi trước của file này để mức 🔴 và ghi p99 = 2.681. Sai ở hai
> chỗ — p99 thật là **2.504** (số cũ lệch do cách nội suy phân vị), và tần suất chỉ 0,3%
> nên mức đúng là 🟡. Hội đồng sẽ hỏi *"hỏng bao nhiêu phần trăm?"*; phải trả lời được.

**Cần sửa** (`AI_service/config.py`):

| Khoá | Hiện tại | Đề xuất | Lý do |
|---|---:|---:|---|
| `solve_full` | 3.500 | **5.500** | đề khó cần tới **4.691** token nghĩ → đang bị cắt |
| `concept_lookup` | 1.500 | **2.500** | dưới p99 tổng thể (2.051) |
| `arena_generation` | 2.000 | **2.500** | dưới p99 tổng thể |
| `generate_mcq` | 2.500 | **3.000** | sát p99 ra đề (2.192), thiếu biên |
| `solve_hint` | 2.500 | 2.500 | giữ — hint ngắn, không cần nhiều |
| `explain_mistake` | 2.500 | **3.000** | sát p99, và là tác vụ đối chiếu (xem mục 3) |
| `grade_essay` | 4.000 | 4.000 | giữ — chấm bài chỉ cần p99 = 626, rất dư |
| `default` | 2.500 | **3.000** | sát p99 |

> Ghi chú: `grade_essay` đang thừa rất nhiều (4.000 cho nhu cầu 626). Không cần hạ —
> ngân sách thừa không tốn gì nếu model không dùng tới — nhưng biết để không tưởng là
> đang thiếu.

---

## 3. 🟡 RAG làm HỎNG các tác vụ ĐỐI CHIẾU — cân nhắc tắt cho `explain_mistake`

**Đo được** — bật RAG làm giảm độ chính xác chấm bài ở **4/5 model**:

| Model | Accuracy thuần → RAG | False-Pass thuần → RAG | |
|---|---:|---:|---|
| Qwen2.5-7B | 79.9% → **74.1%** | 22.1% → **46.3%** | ↓ hại |
| R1-Distill-7B | 75.5% → **68.3%** | 33.8% → **47.8%** | ↓ hại |
| R1-Distill-14B | 86.3% → **80.6%** | 20.6% → 30.9% | ↓ hại |
| **R1-Distill-32B** | **92.1% → 87.0%** | **14.7% → 25.0%** | **↓ hại** |
| Llama-3.1-8B | 74.8% → 78.4% | 30.8% → 27.9% | ↑ **lợi — ngoại lệ** |

**Cơ chế:** ngữ cảnh RAG chứa **bài giải mẫu tương tự** từ ngân hàng câu hỏi. Khi chấm,
model đem bài của sinh viên đối chiếu nhầm với *bài mẫu tương tự* thay vì với **đáp án
chuẩn của chính câu đó** → dễ dãi hẳn, cho bài sai đậu gấp đôi.

> ⚠️ **Đính chính so với bản ghi trước.** File này từng ghi *"cả 5/5 model, không ngoại
> lệ"* — **sai**. Llama-3.1-8B **tăng** 74.8% → 78.4% khi bật RAG. Con số "5/5" bị viết
> cứng trong `report.py` thay vì đếm từ dữ liệu nên không ai phát hiện; nay đã sửa để
> tính động.
>
> Ngoại lệ này **đáng chú ý chứ không đáng giấu**: Llama là model **duy nhất ngoài họ
> Qwen**, nên hiệu ứng "RAG hại tác vụ đối chiếu" có thể gắn với dòng Qwen (cả 4 model
> còn lại đều nền Qwen). Nhưng đừng kết luận vội — Llama cũng là model **yếu nhất**
> (V2-khó chỉ 43.9%), nên còn nhiều chỗ để RAG cứu; và với n=139 thì +3.6 điểm nằm
> trong khoảng nhiễu.
>
> **Kết luận nghiệp vụ không đổi:** model production (32B) **bị RAG làm hại rõ rệt**
> (−5.1 điểm, False-Pass gần gấp đôi) → vẫn không bật RAG cho chấm bài.

**Hiện trạng:**
- `grade_essay` — ✅ **KHÔNG dùng RAG** (đã đúng, không cần sửa)
- `explain_mistake` (dòng ~540) — ❌ **có** `rag.retrieve(query, k=4, ...)`
- `concept_lookup` (dòng ~562) — có `rag.retrieve`, nhưng đây là **tra cứu khái niệm**
  (model THIẾU kiến thức) nên RAG đúng chỗ → **giữ nguyên**

**Cần làm:** `explain_mistake` là tác vụ *đối chiếu bài sai của SV với đáp án* — cùng bản
chất với chấm bài, nhiều khả năng dính đúng cơ chế trên. **Cần đo riêng trước khi tắt**,
đừng suy diễn từ V3 sang.

> Nguyên tắc rút ra: **RAG không phải luôn tốt.** Nó giúp khi model THIẾU kiến thức
> (ra đề, giải toán khó, tra khái niệm), nhưng gây hại khi nhiệm vụ là ĐỐI CHIẾU hai
> thứ đã có sẵn trong prompt — lúc đó tài liệu thêm vào chỉ là nhiễu.

---

## 4. 🟡 Bóc đáp án cuối phải chịu được `\boxed{}` và markdown

**Đo được:** luật "lấy dòng ngay sau marker" hỏng **128/180 câu** với R1-7B (nhưng chỉ
2/180 với Qwen) — vì R1 được huấn luyện trả lời theo markdown + LaTeX:

```
**ĐÁP ÁN CUỐI:**
\[
f(2) = \boxed{3} \quad ; \quad x = \boxed{1} \text{ và } x = \boxed{0.5}
\]
```

Đáp án nằm ở **dòng sau**, trong `\boxed{}`. Bóc kiểu cũ ra `**` hoặc `\[`.

**Cần làm:** nếu AI service có chỗ nào bóc đáp án cuối từ text tự do, dùng lại
`bench/tasks.py::extract_final()` trong `harness/` — nó ưu tiên `\boxed{}` (có đếm ngoặc
lồng để xử lý `\boxed{\frac{1}{2}}`), rồi mới tới dòng đầu tiên có nội dung. Đã kiểm:
sửa 128/180 → 0/180 hỏng.

> Lưu ý: 32B **không** dùng `\boxed{}` mà dùng `\( \)`. Bộ bóc phải xử lý cả hai —
> chỉ bắt `\boxed{}` thì hỏng toàn bộ 32B.

---

## 5. 🟢 Tài liệu MAD101 bị trùng trong `document_chunks`

**Đo được:** `document_chunks` có cả tài liệu cũ lẫn mới cho MAD101:

```
MAD101  "Đồ thị"                    187 chunk
MAD101  "Đồ Thị"                     66 chunk   ← bản cũ, trùng nội dung
MAD101  "Logic mệnh đề & vị từ"     205 chunk
MAD101  "Logic"                      65 chunk   ← bản cũ
MAD101  "Quy nạp & đệ quy"           92 / "Quy Nạp & Đệ Quy" 54
MAD101  "Số học mô-đun & mật mã"     74 / "Số Học Mô - Đun & Mật Mã" 51
MAD101  "Tập hợp, Hàm số, Dãy số"   142 / "Tập Hợp, Hàm Số, Dãy Số" 48
MAD101  "Thuật toán & đô phức tạp"  165 / "Thuật Toán Và Độ Phức Tạp" 38
MAD101  "Đếm & hệ thức truy hồi"     46 / "Đếm & Hệ Thức Truy Hồi" 30
```

405 chunk cũ (8 tài liệu, embed đợt đầu) chưa xoá khi nạp bản mới → tổng MAD101 = 1.427.

**Hậu quả:** top-3 truy hồi có thể trả về 2 đoạn gần trùng nhau → phí chỗ trong context,
model nhận ít thông tin hữu ích hơn.

**Cần làm:** xoá 8 tài liệu cũ (tên viết hoa kiểu Title Case). Không gấp — benchmark vẫn
đạt 100% truy hồi được lý thuyết cho cả 3 môn.

---

## 6. 🟢 Độ phủ lý thuyết lệch giữa các môn

```
MAD101  1.427 chunk     ← nhiều nhất
MAE101    550 chunk
MAS291    198 chunk     ← ít nhất, kém MAD 7 lần
```

Không phải lỗi code. Nhưng môn ít tài liệu thì RAG có ít thứ để lấy → là nguồn gốc của
phần lớn *Retrieval Error*. Nếu muốn cải thiện chất lượng MAS291 thì bổ sung tài liệu là
đòn bẩy lớn hơn đổi model.

---

## 7. 🟢 Chat / Đấu trường: 32B quá chậm

**Đo được:** độ trễ P50 ở concurrency=1 (số chốt sau khi chạy đủ 5 model)

| Model | Độ trễ P50 | Điện năng/câu |
|---|---:|---:|
| Qwen2.5-7B | **3.3s** | 0.38 Wh |
| Llama-3.1-8B | 3.5s | 0.39 Wh |
| R1-Distill-7B | 4.9s | 0.60 Wh |
| R1-Distill-14B | 12.0s | 1.38 Wh |
| **R1-Distill-32B** | **34.7s** | **4.16 Wh** |

32B chậm hơn Qwen **10.5×** và tốn điện **11×**. Với Chat/Đấu trường (SV chờ realtime)
thì 34.7s là không dùng được.

> Đính chính: bản ghi trước để 42s / 8.6s / ~17s — đó là số của lượt chạy nháp trước khi
> chốt cấu hình (32B khi đó còn chạy `--enforce-eager`, bị bóp ~10-15%). Số trong bảng
> này lấy từ `results/summary.json` của lượt chạy cuối, khớp với `BAO_CAO_AI.md`.

**Vấn đề:** 96GB **không** chở nổi 32B + một model nhanh cùng lúc — đo thật: 32B chiếm
87.3GB, còn 6.1GB, mà model 7B cần tối thiểu 14.25GB chỉ riêng trọng số.

**Hướng chưa kiểm chứng:** 32B-AWQ (4-bit, ~20GB) đủ chỗ chạy kèm 7B trên cùng 1 GPU.
**Nhóm CHƯA đo** — không được đưa vào kết luận, nhưng đáng thử nếu cần cả nhanh lẫn sâu.

---

## 8. 📌 Ghi nhận (không phải lỗi): Llama-3.1-8B yếu toán THẬT, không phải giám khảo chấm oan

Không cần sửa gì trong service — ghi lại vì đây là chỗ **hội đồng chắc chắn vặn**:
*"Llama thấp thế kia, có phải do giám khảo là model họ DeepSeek nên thiên vị không?"*

**Số đo:**

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---:|---:|
| V2 bộ gốc (thuần) | 94.4% | **71.7%** |
| V2 bộ KHÓ (thuần) | 73.3% | **43.9%** |
| V3 chấm bài (thuần) | 79.9% | 74.8% |

**Đã kiểm chứng KHÔNG phải do giám khảo:**

1. **Kiểm độc lập bằng code, không qua LLM.** So chuỗi đáp án cuối với `expected_answer`
   trực tiếp — Llama vẫn thấp hẳn. Có soi tay 2 ca giám khảo chấm sai rõ ràng; sửa cả 2
   thì Llama lên 72.8%, **vẫn cách Qwen ~22 điểm** ở bộ khó.
2. **Khớp với benchmark đã công bố.** Qwen2.5 được huấn luyện nặng về toán (GSM8K/MATH),
   Llama-3.1-8B là model đa dụng — chênh lệch toán giữa hai dòng này là điều đã biết.
3. **V3 dùng nhãn do code sinh, không nhờ giám khảo.** Ở đó Llama đạt 74.8% — *không*
   sụp đổ. Nếu giám khảo thiên vị họ model thì V3 cũng phải thấp; nó không thấp.

**⚠️ Đính chính một khẳng định tôi từng đưa ra:** tôi từng nói *"hai baseline hội tụ
(79.9 / 80.0)"* ở V3 — **sai**. Con số 80.0 tính trên mẫu số đã lọc (chỉ các dòng có
phán quyết). Tính trên **toàn bộ** 139 mẫu thì Llama là **74.8%**, kém Qwen 79.9% khoảng
5 điểm — vì Llama **không in ra nhãn phán quyết ở ~10% số ca** (`format_ok` thấp nhất
nhóm, truncation 3.8% cao gấp ~10× các model khác). Không in được nhãn **cũng là hỏng**,
không được loại khỏi mẫu số.

> Bài học: **cẩn thận với mẫu số.** Lọc bỏ ca hỏng rồi tính tỉ lệ trên phần còn lại sẽ
> "tặng điểm" đúng cho model hay hỏng nhất. Mẫu số phải là **toàn bộ số ca đã giao**.

---

## Việc cho vòng sau: thêm Vấn đề 4 — đánh giá năng lực toán của SV

Harness giữ lại ở `harness/` chính vì việc này. Cách thêm:

1. `bench/tasks.py` — thêm `build_v4(...)` trả `Task(problem="V4", ...)`
2. `bench/run.py` — `build_all_tasks()` gộp thêm v4
3. `bench/judge.py` — thêm nhánh xử lý `"V4"` (hoặc chấm bằng code nếu có nhãn thật)
4. `bench/analyze.py` — thêm `"V4"` vào vòng lặp `for prob in (...)`
5. `bench/report.py` — thêm dòng vào bảng chất lượng

**Cảnh báo thiết kế quan trọng:** đừng để LLM tự chấm điểm năng lực. Điểm năng lực phải
tính bằng code (IRT / Elo / tỉ lệ đúng theo chủ đề); LLM chỉ dùng để *diễn giải* điểm đó
thành nhận xét. Lý do: bài học từ chính benchmark này — nhãn do code sinh cho ra số liệu
**không ai vặn được** (V3), còn số liệu qua giám khảo LLM thì luôn phải kèm phần hiệu
chuẩn và giải trình.
