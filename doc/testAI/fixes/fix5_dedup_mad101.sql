-- ============================================================================
-- Fix #5 (benchmark 15/07): xoá tài liệu MAD101 bị TRÙNG trong document_chunks
--
-- Nền tảng: 405 chunk cũ (8 tài liệu, embed đợt đầu, tên viết hoa kiểu Title Case)
-- chưa được xoá khi nạp bản mới → top-3 truy hồi có thể trả 2 đoạn gần trùng nhau.
--
-- ⚠️ CHẠY TRÊN DB CỦA AI SERVICE (Postgres + pgvector), KHÔNG phải DB của BE.
-- ⚠️ CHẠY TỪNG BƯỚC. Bước 1–2 chỉ ĐỌC. Chỉ chạy bước 3 sau khi đã xem kỹ bước 1.
-- ⚠️ SAO LƯU trước: pg_dump -t documents -t document_chunks <db> > backup_docs.sql
--
-- Chunk sẽ tự xoá theo document nhờ: document_chunks.document_id ... ON DELETE CASCADE.
-- ============================================================================

-- ── Bước 1. XEM toàn bộ tài liệu MAD101 kèm số chunk + thời điểm tạo ─────────
-- Nhìn cột title: các bản cũ có tên Title Case ("Đồ Thị", "Logic",
-- "Quy Nạp & Đệ Quy"...) và created_at SỚM hơn bản mới. Ghi lại id của chúng.
SELECT d.id,
       d.title,
       d.created_at,
       count(c.id) AS chunks
FROM documents d
LEFT JOIN document_chunks c ON c.document_id = d.id
WHERE d.subject = 'MAD101' OR d.subject = 'MAD'
GROUP BY d.id, d.title, d.created_at
ORDER BY lower(d.title), d.created_at;

-- ── Bước 2. TỰ ĐỘNG phát hiện trùng do KHÁC HOA/THƯỜNG (an toàn tuyệt đối) ───
-- Nhóm theo tên đã chuẩn hoá (bỏ dấu cách thừa + lower). Nhóm nào có >1 bản thì
-- bản created_at SỚM NHẤT là bản cần xoá. Đây chỉ là câu SELECT để xem trước.
WITH norm AS (
    SELECT id, title, created_at,
           lower(regexp_replace(trim(title), '\s+', ' ', 'g')) AS key
    FROM documents
    WHERE subject IN ('MAD101', 'MAD')
),
ranked AS (
    SELECT id, title, created_at, key,
           row_number() OVER (PARTITION BY key ORDER BY created_at DESC) AS rn
    FROM norm
)
SELECT id, title, created_at, key, rn,
       CASE WHEN rn > 1 THEN 'XOÁ (bản cũ hơn)' ELSE 'GIỮ (mới nhất)' END AS action
FROM ranked
WHERE key IN (SELECT key FROM ranked GROUP BY key HAVING count(*) > 1)
ORDER BY key, created_at DESC;

-- ── Bước 3a. XOÁ tự động các bản trùng do khác hoa/thường ────────────────────
-- Bỏ chú thích để chạy SAU KHI đã xem bước 2 và đồng ý.
-- (Chunk tự xoá theo CASCADE.)
/*
WITH norm AS (
    SELECT id, created_at,
           lower(regexp_replace(trim(title), '\s+', ' ', 'g')) AS key
    FROM documents
    WHERE subject IN ('MAD101', 'MAD')
),
ranked AS (
    SELECT id, row_number() OVER (PARTITION BY key ORDER BY created_at DESC) AS rn
    FROM norm
)
DELETE FROM documents
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);
*/

-- ── Bước 3b. XOÁ theo DANH SÁCH ID thủ công (cho các bản trùng NGỮ NGHĨA) ────
-- Trường hợp "Logic" vs "Logic mệnh đề & vị từ" — khác tên nên bước 2 KHÔNG bắt.
-- Sau khi xem bước 1, điền id các bản cũ vào đây rồi bỏ chú thích.
/*
DELETE FROM documents
WHERE id IN (
    -- 'doc_id_cu_1',
    -- 'doc_id_cu_2',
    -- ...
);
*/

-- ── Bước 4. KIỂM TRA sau khi xoá ────────────────────────────────────────────
-- Tổng chunk MAD101 nên giảm ~405 (từ 1.427 về ~1.022).
-- SELECT count(*) FROM document_chunks WHERE subject IN ('MAD101','MAD');
