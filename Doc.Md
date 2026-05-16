# Tài Liệu Phạm Vi Đồ Án FPTU_MATHAI

## 1. Mục Tiêu Đồ Án

FPTU_MATHAI là hệ thống học toán cho sinh viên FPT, hỗ trợ ba nhóm người dùng chính:

- `Student`: học, làm bài, nộp bài, xem kết quả, hỏi AI.
- `Teacher`: quản lý lớp, tạo bài, chấm bài, xem phân tích học lực.
- `Admin`: quản lý kỳ học, môn học, lớp học, tài khoản và dữ liệu hệ thống.

Định hướng triển khai là để phần SE chiếm phần lớn khối lượng đồ án, còn phần AI tập trung vào các chức năng thật sự cần AI:

- Sinh đề toán.
- Giải toán từng bước.
- Chấm bài tự luận.
- Giải thích lỗi sai.
- Tự đánh giá năng lực.
- Phân tích điểm yếu của lớp.
- Chatbox AI cho giáo viên.

## 2. Nguyên Tắc Chia Scope

Nhóm có 1 thành viên AI và 4 thành viên SE. Vì vậy:

- AI làm phần model, data, vector DB, prompt, inference và các API/service AI.
- SE làm phần hệ thống lớp học, user, role, bài tập, bài kiểm tra, video, tài liệu, dashboard, notification, chat, quản trị.
- AI không nên ôm toàn bộ nghiệp vụ giáo dục. AI chỉ trả kết quả đề xuất, hệ thống SE chịu trách nhiệm workflow, phân quyền, lưu trạng thái và review.

## 3. Scope AI Cốt Lõi

### 3.1 AI Sinh Bài Tập

Giáo viên nhập:

- Môn học
- Chương
- Topic
- Độ khó: easy, medium, hard
- Số lượng câu hỏi

AI trả về câu hỏi theo schema giống `Data_Bank.json`, gồm:

- Đề bài
- LaTeX nếu có
- Đáp án cuối
- Lời giải từng bước
- Hint
- Common mistakes
- Concepts used
- Metadata

Câu hỏi được lưu vào PostgreSQL và vector hóa để search.

### 3.2 AI Giải Toán Chuyên Dụng FPT

Sinh viên có thể gửi:

- Text đề bài
- Ảnh chụp đề bài
- OCR text từ backend

AI giải từng bước và lưu lại đề cùng lời giải vào database. Với ảnh, có thể dùng OCR trước rồi đưa text vào LLM để ổn định hơn.

### 3.3 AI Chấm Bài Tự Luận

AI nhận:

- Đề bài
- Bài làm sinh viên
- Rubric nếu giáo viên cung cấp

AI trả:

- `DUNG` hoặc `SAI`
- Điểm đề xuất
- Feedback
- Lỗi sai
- Nhận xét từng bước
- Gợi ý sửa

Giáo viên vẫn có quyền override điểm.

### 3.4 AI Giải Thích Câu Sai

Sau khi sinh viên nộp bài, nếu câu trả lời sai, AI giải thích:

- Sai ở đâu
- Vì sao sai
- Cần ôn lại khái niệm nào
- Cách sửa
- Hướng làm đúng

### 3.5 AI Tự Kiểm Tra Năng Lực

AI tạo bộ câu hỏi diagnostic theo môn hoặc topic. Sau khi sinh viên trả lời, AI đánh giá:

- Level hiện tại
- Điểm mạnh
- Điểm yếu
- Lộ trình học đề xuất
- Topic nên luyện thêm

### 3.6 AI Phân Tích Điểm Yếu Của Lớp

AI nhận records điểm số, submission, topic, mistake và phân tích:

- Chương nào sai nhiều nhất
- Topic nào yếu nhất
- Dạng bài nào cần ôn
- Sinh viên nào cần hỗ trợ
- Gợi ý kế hoạch ôn tập

### 3.7 AI Chatbox Cho Giáo Viên

Giáo viên có thể hỏi bằng ngôn ngữ tự nhiên:

- Lớp này yếu chương nào?
- Nên giao thêm dạng bài gì?
- Sinh viên nào có nguy cơ rớt?
- Hãy đề xuất kế hoạch ôn tập 2 tuần.

AI dùng dữ liệu lớp và question bank để trả lời.

## 4. Scope SE Chính

Phần SE nên chiếm phần lớn đồ án, gồm các module sau.

### 4.1 Student - Lớp Học

1. Dashboard lớp học.
2. Xem danh sách bài tập được giao.
3. Nộp bài tập bằng file PDF, ảnh hoặc text.
4. Xem trạng thái nộp bài và lịch sử nộp lại.
5. Làm bài kiểm tra trắc nghiệm và tự luận.
6. Đồng hồ đếm ngược khi thi.
7. Tự động nộp bài khi hết giờ.
8. Anti-cheat: fullscreen, tab switch log, cảnh báo.
9. Xem kết quả bài thi hoặc bài tập.
10. AI giải thích từng câu sai sau khi nộp.
11. Biểu đồ xu hướng điểm số.
12. Chat real-time với giáo viên.
13. Gọi video với giáo viên.
14. Chat với AI.
15. Lịch sử hội thoại AI.
16. Calendar deadline và lịch thi.
17. Notification real-time.
18. Email notification.
19. Báo cáo câu hỏi lỗi hoặc không rõ.

### 4.2 Student - Tự Học

1. Chọn môn học muốn tự học.
2. Chọn giáo viên giảng dạy.
3. Lưu tiến trình theo cặp môn học và giáo viên.
4. Lưu tối đa 3 tiến trình cũ.
5. Xem video bài giảng YouTube unlisted.
6. Resume video từ thời điểm bỏ dở.
7. Điều chỉnh tốc độ phát video.
8. Ghi nhận video đã xem.
9. Xem slide và giáo trình PDF inline.
10. Download tài liệu.
11. Tìm kiếm nội dung tài liệu.
12. Dashboard tiến trình học.
13. Tự kiểm tra năng lực bằng AI.
14. Luyện tập theo chủ đề và độ khó.
15. Ghi chú cá nhân theo video hoặc chương.
16. Lịch sử bài tập đã làm.
17. Huy hiệu và thành tích.

### 4.3 Student - Profile

1. Trang hồ sơ cá nhân.
2. Avatar và thông tin sinh viên.
3. Lớp đang học.
4. Huy hiệu.
5. Export báo cáo học tập PDF theo kỳ.

### 4.4 Teacher - Quản Lý Lớp

1. Dashboard lớp.
2. Điểm trung bình, pass/fail, histogram.
3. Top 5 và bottom 5 sinh viên.
4. Danh sách sinh viên, filter và sort.
5. Xem lịch sử từng sinh viên.
6. Điểm danh theo buổi.
7. Báo cáo chuyên cần.
8. Export Excel.
9. Đăng thông báo cho lớp.
10. Chat real-time với sinh viên.
11. Gọi video với sinh viên.

### 4.5 Teacher - Tạo Nội Dung

1. AI sinh câu hỏi tự động.
2. Giáo viên review câu hỏi AI sinh.
3. Ngân hàng câu hỏi CRUD.
4. Tag câu hỏi theo chương, topic, độ khó.
5. Search câu hỏi bằng keyword và vector.
6. Import Excel câu hỏi.
7. Preview LaTeX.
8. Tạo bài tập từ ngân hàng câu hỏi.
9. Tạo bài kiểm tra từ ngân hàng câu hỏi.
10. Chọn câu hỏi thủ công hoặc random theo tag.
11. Cấu hình deadline.
12. Cấu hình thời gian làm bài.
13. Cấu hình số lần retry.
14. Trộn đề tự động.
15. Template bài tập cho kỳ sau.
16. Quản lý video bài giảng.
17. Upload slide, giáo trình, tài liệu.

### 4.6 Teacher - Chấm Điểm Và Phân Tích

1. AI chấm bài tự luận.
2. Giáo viên override điểm AI.
3. Ghi chú lý do override.
4. Lịch sử chỉnh sửa điểm.
5. Export bảng điểm Excel hoặc PDF.
6. Xếp loại sinh viên A/B/C/D/F.
7. AI phân tích điểm yếu sinh viên.
8. AI phân tích điểm yếu của lớp.
9. AI chatbox cho giáo viên.

### 4.7 Admin - Quản Lý Hệ Thống

1. Quản lý kỳ học.
2. Quản lý môn học.
3. Gán môn vào kỳ.
4. Cấu hình curriculum theo chương hoặc tuần.
5. Tạo lớp học.
6. Gán môn, kỳ và giáo viên cho lớp.
7. Thêm sinh viên vào lớp.
8. Bulk import sinh viên theo khóa.
9. Quản lý tài khoản bằng email hoặc FEID.
10. Phân loại khóa K18B, K18C.
11. Gán role.
12. Kích hoạt hoặc vô hiệu hóa tài khoản.
13. Import Excel sinh viên và giáo viên.
14. Dashboard hệ thống.
15. Thống kê lượt dùng AI.
16. Broadcast thông báo.
17. Quản lý câu hỏi bị report.
18. Audit log hành động quan trọng.

## 5. Data Và Question Bank

Hệ thống cần có bộ dữ liệu lớn về:

- Đề toán.
- Lời giải toán.
- Hint.
- Common mistakes.
- Topic, subtopic, chapter.
- Độ khó.
- Dạng bài.
- Rubric chấm điểm.

`Data_Bank.json` là seed data ban đầu. Các câu hỏi mới có thể đến từ:

- Đề thật được nhập thủ công.
- Đề AI sinh và giáo viên review.
- Đề sinh viên upload.
- Dataset mở được làm sạch và chuyển schema.

Tất cả câu hỏi nên được đưa vào PostgreSQL với vector embedding để phục vụ search và RAG.

## 6. AI Service Hiện Tại

Code AI đã được tách vào folder `AI_service`.

Các command:

```powershell
python -m AI_service.main generate
python -m AI_service.main solve-upload
python -m AI_service.main evaluate-answer
python -m AI_service.main explain-wrong
python -m AI_service.main self-assess
python -m AI_service.main analyze-class
python -m AI_service.main teacher-chat
python -m AI_service.main classify-question
python -m AI_service.main import-json
python -m AI_service.main search
```

## 7. Mapping Feature AI Với Command

| Feature | Command |
| --- | --- |
| AI ra bài tập theo chương và độ khó | `generate` |
| AI giải toán chuyên dụng FPT | `solve-upload` |
| AI chấm bài tự luận | `evaluate-answer` |
| AI giải thích câu sai | `explain-wrong` |
| AI tự đánh giá năng lực | `self-assess` |
| AI phân tích điểm yếu lớp | `analyze-class` |
| AI chatbox cho giáo viên | `teacher-chat` |
| AI phân loại bài toán | `classify-question` |
| Lưu ngân hàng đề dạng vector | `import-json`, `generate`, `solve-upload` |
| Search câu hỏi bằng vector | `search` |

## 8. Rủi Ro Và Giới Hạn

- LLM có thể giải sai toán, nên cần giáo viên review câu hỏi và lời giải quan trọng.
- OCR ảnh toán có thể sai ký hiệu, nên backend nên cho user sửa text OCR trước khi gửi AI.
- Dataset hiện chưa đủ lớn, cần bổ sung thêm đề thật.
- AI chấm tự luận nên là điểm đề xuất, không nên là điểm cuối cùng nếu chưa có giáo viên xác nhận.
- Fine-tune LoRA cần GPU phù hợp. Nếu máy local yếu, nên train trên Colab hoặc server GPU.

## 9. Kết Luận Scope

Phần AI đủ để chứng minh đồ án có nghiên cứu và ứng dụng AI riêng:

- Fine-tune DeepSeek bằng LoRA.
- Xây dựng dữ liệu SFT từ question bank.
- Lưu question bank dạng vector.
- Sinh đề và giải toán.
- Chấm bài và giải thích lỗi sai.
- Phân tích lớp bằng AI.
- Tích hợp được với backend thông qua CLI hoặc module Python.

Phần còn lại như lớp học, bài kiểm tra, video, tài liệu, chat, notification, dashboard và admin panel thuộc scope SE để bảo đảm đồ án đủ lớn.
