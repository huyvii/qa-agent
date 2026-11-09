# QA Agent — Hỏi đáp tài liệu PDF bằng Hybrid RAG

Ứng dụng hỏi đáp trên nhiều tài liệu PDF dựa trên kiến trúc Retrieval-Augmented Generation (RAG) hai giai đoạn: kết hợp tìm kiếm ngữ nghĩa đa ngữ (dense vector) với tìm kiếm từ khóa (Lucene BM25), hợp nhất bằng Reciprocal Rank Fusion (RRF), xếp hạng lại bằng Cross-Encoder đa ngữ cục bộ, rồi sinh câu trả lời kèm trích dẫn trang bằng Gemini.

## Tính năng chính

- **Quản lý đa tài liệu & đa hội thoại:** Tải lên nhiều PDF (tối đa 30 MB/file), tạo nhiều cuộc trò chuyện độc lập trên từng tài liệu, tự đặt tiêu đề theo câu hỏi đầu tiên và lọc truy hồi đúng theo `source_file`.
- **Hybrid Retrieval + Cross-Encoder Reranking:** Kết hợp `paraphrase-multilingual-MiniLM-L12-v2` và BM25 (biến thể IDF chuẩn Lucene luôn dương trên tập nhỏ, có cache theo từng tài liệu), hợp nhất top ứng viên bằng RRF rồi chấm điểm lại bằng `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- **Bộ nhớ hội thoại cục bộ (1 lượt gọi LLM/câu):** Lưu toàn bộ lịch sử trong SQLite; mỗi lượt trả lời ghép 4 lượt gần nhất và tối đa 2 lượt cũ liên quan (truy hồi bằng BM25 trong cửa sổ 60 lượt trước). Câu hỏi nối tiếp được mở rộng truy vấn cục bộ theo câu hỏi mỏ neo gần nhất mà không tốn thêm request Gemini.
- **Định danh tài liệu an toàn bằng SHA-256:** Bỏ qua không nhúng lại nếu nội dung không đổi; băm tất định ID từng chunk để chống nhân bản vector; khi cập nhật file trùng tên chỉ xóa vector phiên bản cũ sau khi phiên bản mới đã lập chỉ mục thành công.
- **Trích dẫn trung thực & kiểm soát lỗi:** Chuyển số trang về chuẩn 1-based, khử trùng lặp trang trích dẫn, tự ẩn nguồn PDF khi câu trả lời nằm ngoài phạm vi tài liệu, cảnh báo trang scan ít văn bản và phân loại rõ lỗi API (`401/403`, `429`, `404`).
- **Đồng bộ giao diện Web & CLI:** Hỗ trợ cả giao diện Streamlit (`app.py`) lẫn dòng lệnh (`ingest.py`, `query.py`) dùng chung kho ChromaDB và bảng `documents` trong SQLite.

## Kiến trúc & Luồng xử lý

```text
Ingest:
  PDF ──► SHA-256 ──► PyPDFLoader ──► RecursiveCharacterTextSplitter (500 / 50)
      ──► Multilingual MiniLM Embedding ──► ChromaDB (kèm source_file, page, content_hash)
      ──► SQLite documents (metadata & kiểm tra trùng lặp)

Query (1 lượt gọi Gemini duy nhất):
  Câu hỏi + Lịch sử SQLite
      ├─► Mở rộng câu nối tiếp (giữ câu hỏi mỏ neo) ──► Truy vấn retrieval
      └─► Chọn ngữ cảnh hội thoại (4 lượt gần nhất + tối đa 2 lượt cũ qua Lucene BM25)
                                  │
                                  ▼
      ┌── Nhánh Vector (Chroma similarity_search, lọc source_file) ──┐
      │                                                              ├──► RRF (pool=10)
      └── Nhánh Từ khóa (Lucene BM25 có cache theo source_file) ─────┘         │
                                                                               ▼
                                                     Cross-Encoder Reranker đa ngữ (top-3)
                                                                               │
                                                                               ▼
                                              Gemini (streaming) ──► Câu trả lời + Nguồn trang
```

| Tệp | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit, quản lý tài liệu/cuộc trò chuyện, hiển thị latency và xử lý lỗi |
| `rag_core.py` | Pipeline nạp PDF, cache Lucene BM25, hợp nhất RRF, Cross-Encoder reranking, bộ nhớ hội thoại và gọi Gemini |
| `chat_db.py` | Lưu trữ bền vững tài liệu, cuộc trò chuyện và tin nhắn trên SQLite |
| `ingest.py` | Công cụ CLI nạp PDF (đồng bộ SHA-256 và metadata với SQLite/UI) |
| `query.py` | Công cụ CLI đặt câu hỏi trực tiếp từ terminal (`--no-rerank` nếu cần phản hồi nhanh) |
| `evaluation/retrieval_baseline.py` | Benchmark thực nghiệm so sánh các cấu hình retrieval (`Hit@1`, `Hit@k`, `MRR@k`, độ trễ) |
| `tests/` | Bộ 29 unit test kiểm thử SQLite, BM25, RRF, reranker, bộ nhớ hội thoại và chuẩn hóa eval |

## Cài đặt

Yêu cầu Python 3.10 trở lên và [Gemini API key](https://aistudio.google.com/apikey).

```powershell
git clone https://github.com/huyvii/qa-agent.git
cd qa-agent
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Mở `.env` và điền khóa API của bạn:

```ini
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
GEMINI_MAX_RETRIES=2
```

`GEMINI_MODEL` và `GEMINI_MAX_RETRIES` là tùy chọn. Không commit `.env`, `chroma_db/` hoặc `chat_history.db`. Khi triển khai trên Streamlit Community Cloud, khai báo các biến này trong mục **Secrets**.

## Sử dụng

### 1. Giao diện Web (Streamlit)

```powershell
streamlit run app.py
```

- Tải file PDF ở thanh bên để lập chỉ mục và mở cuộc trò chuyện đầu tiên.
- Với tài liệu đã có trong mục **Tài liệu đã nạp**, nhấn `➕` để tạo cuộc trò chuyện mới ngay mà không cần tải lại file, hoặc `🗑️` để xóa tài liệu kèm xác nhận.
- Lần chạy đầu tiên sẽ tải model embedding (~470 MB) và reranker (~470 MB) về cache HuggingFace cục bộ.

### 2. Dòng lệnh (CLI)

```powershell
python ingest.py .\path\to\document.pdf
python query.py "Tóm tắt nội dung chính" --source-file document.pdf
python query.py "Công thức chính ở mục 2 là gì?" --source-file document.pdf --no-rerank
```

## Đánh giá thực nghiệm Retrieval (Benchmark)

Bộ đánh giá tự chứa trong `tests/fixtures/math_exercises.md` gồm **24 đoạn văn học thuật song ngữ Việt – Anh** (Đại số tuyến tính, Xác suất Bayes, Tối ưu hóa AdamW/Lipschitz, Attention, LoRA, FlashAttention, RAG/RRF — có các đoạn nhiễu cùng chủ đề) và **12 câu hỏi tiếng Việt** chia thành 4 nhóm: khớp công thức/tham số chính xác, phân biệt đoạn nhiễu, diễn giải khái niệm và truy vấn chéo ngôn ngữ Việt → Anh.

Chạy đối chiếu cả 4 cấu hình trên cùng một tập chỉ mục (không gọi Gemini API):

```powershell
python evaluation/retrieval_baseline.py --compare
```

Kết quả đo thực tế trên CPU (`top-k = 3`, trung bình sau bước làm nóng cache):

| Cấu hình | Hit@1 | Hit@3 | MRR@3 | Độ trễ TB (CPU) |
|---|---:|---:|---:|---:|
| BM25 (Lucene IDF, lexical) | 8/12 (67%) | 8/12 (67%) | 0.667 | ~0.3 ms |
| Multilingual Vector (`paraphrase-multilingual-MiniLM-L12-v2`) | 10/12 (83%) | 11/12 (92%) | 0.875 | ~27 ms |
| Hybrid: Vector + BM25, hợp nhất RRF (`RRF_K = 60`) | 7/12 (58%) | 10/12 (83%) | 0.681 | ~29 ms |
| **Hybrid + Cross-Encoder (`mmarco-mMiniLMv2-L12-H384-v1`)** | **11/12 (92%)** | **12/12 (100%)** | **0.958** | **~331 ms** |

**Phân tích kỹ thuật từ thực nghiệm:**
- **Bù trừ giữa BM25 và Vector:** BM25 bắt chính xác tuyệt đối các hằng số/ký hiệu (`k_rrf = 60` đứng hạng 1 trong khi Vector trượt hoàn toàn), nhưng đạt `0/4` ở các câu hỏi tiếng Việt tra cứu đoạn tiếng Anh không trùng từ vựng. Ngược lại, Multilingual Vector giải quyết được cả `4/4` câu chéo ngôn ngữ Việt → Anh.
- **Tại sao RRF thuần làm giảm `Hit@1` nhưng lại cần thiết cho giai đoạn 1:** Với `RRF_K = 60`, một đoạn tiếng Việt khớp từ chức năng ở cả hai nhánh nhận xấp xỉ `2 / (60 + r)`, dễ vượt điểm của đoạn tiếng Anh chỉ đứng hạng 1 ở nhánh Vector (`1 / 61`). Tuy nhiên, trong tập ứng viên `RERANK_POOL_SIZE = 10`, RRF gom đủ **12/12** đoạn đúng từ cả hai nhánh.
- **Vai trò của Cross-Encoder đa ngữ ở giai đoạn 2:** Đưa 10 ứng viên của RRF qua mô hình tự chú ý chéo câu hỏi – văn bản (`mmarco-mMiniLMv2-L12-H384-v1`) kéo các đoạn đúng lên đầu bảng xếp hạng (`Hit@3` đạt `12/12`, `MRR@3` đạt `0.958`), đồng thời nhanh hơn ~3.2 lần và nhẹ hơn một nửa RAM so với `BAAI/bge-reranker-base` (vốn chỉ huấn luyện trên tiếng Anh và tiếng Trung, chỉ đạt `10/12` và tốn ~1.07s/câu).

Có thể đánh giá trên kho Chroma thực tế của riêng bạn bằng:

```powershell
python evaluation/retrieval_baseline.py --chroma-dir .\chroma_db --source-file "ten_file.pdf" --rerank
```

## Kiểm thử

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -v
```

## Tốc độ phản hồi, Quota Gemini & Bảo mật

- **Các yếu tố quyết định thời gian phản hồi:** Tổng độ trễ gồm (1) retrieval + Cross-Encoder cục bộ (~0.35s trên CPU sau lần nạp đầu), (2) thời gian chờ token đầu tiên từ Gemini (mạng + bước suy luận nội bộ *thinking* nếu dùng dòng model có thinking), và (3) thời gian sinh tuần tự từng output token (tỷ lệ thuận với độ dài câu trả lời). Ứng dụng dùng streaming để hiển thị chữ ngay khi có token đầu tiên và không gọi LLM lần hai để viết lại câu hỏi.
- **Quota API:** Ứng dụng không tự giới hạn số câu hỏi; hạn mức phụ thuộc vào tier của project trên Google AI Studio (HTTP `429` khi vượt RPM/TPM/RPD).
- **Lưu trữ & quyền riêng tư:** ChromaDB và SQLite lưu cục bộ (đổi vị trí qua `CHROMA_DIR`, `CHAT_DB_PATH` khi gắn volume Docker). Biến `APP_PASSWORD` chỉ cung cấp cổng mật khẩu chung cho bản demo; dữ liệu hiện chia theo cuộc trò chuyện chứ chưa tách biệt theo tài khoản người dùng (multi-tenant).
- **Giới hạn hiện tại:** Pipeline trích xuất văn bản qua `pypdf`; các PDF bản scan thuần ảnh cần OCR trước khi nạp, và bảng biểu nhiều cột hoặc công thức toán phức tạp có thể bị xô lệch thứ tự ký tự khi tách chunk cố định.
