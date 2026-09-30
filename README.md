<h1 align="center">QA Agent</h1>

<p align="center">
  <b>RAG-based PDF Question Answering</b><br>
  Upload PDF, đặt câu hỏi và nhận câu trả lời kèm nguồn tham khảo theo tên file và số trang.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/ChromaDB-vector%20store-FF6446" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Gemini-API-4285F4?logo=googlegemini&logoColor=white" alt="Gemini">
</p>

<p align="center">
  <a href="#tính-năng">Tính năng</a> ·
  <a href="#cách-hoạt-động">Kiến trúc</a> ·
  <a href="#cài-đặt">Cài đặt</a> ·
  <a href="#đánh-giá-retrieval">Đánh giá</a> ·
  <a href="#giới-hạn">Giới hạn</a>
</p>

## Tính năng

- Upload và ingest nhiều file PDF.
- Nhiều cuộc trò chuyện; mỗi cuộc gắn với một PDF. Retrieval lọc theo `source_file` để không lẫn dữ liệu giữa các file.
- Hiển thị nguồn tham khảo gồm tên file và số trang.
- Trả lời stream từng token; câu hỏi nối tiếp được viết lại thành câu độc lập trước khi tra cứu để giữ ngữ cảnh hội thoại (multi-turn).
- Nhận diện tài liệu bằng hash nội dung (SHA-256): cùng file không nhúng lại; trùng tên nhưng khác nội dung tự động xóa vector cũ và nạp lại.
- Trang PDF gần như không có text (bản scan/hình ảnh) được cảnh báo ngay khi ingest.
- Gate mật khẩu tùy chọn qua biến `APP_PASSWORD` — bật khi đưa link cho người khác.
- Xóa cuộc trò chuyện chỉ xóa lịch sử chat, không xóa vector của PDF.
- Ưu tiên trả lời theo PDF; với câu hỏi ngoài phạm vi tài liệu, câu trả lời được đánh dấu rõ là kiến thức chung.

## Cách hoạt động

```text
PDF → tách chunk → embedding multilingual (paraphrase-multilingual-MiniLM-L12-v2)
    → lưu vào ChromaDB kèm metadata (source_file, page)
    → hybrid retrieval: vector + BM25, hợp nhất bằng RRF, lọc theo source_file
    → ghép context vào prompt → Gemini sinh câu trả lời (streaming)
```

Lịch sử và metadata được lưu trong SQLite:

```text
documents       — tài liệu đã ingest
conversations   — mỗi conversation gắn với một source_file
messages        — câu hỏi, câu trả lời và nguồn tham khảo
```

| File | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit |
| `rag_core.py` | Ingest, retrieval và gọi Gemini |
| `chat_db.py` | Quản lý SQLite |

**Công nghệ:** Python, LangChain, ChromaDB, sentence-transformers, BM25 (hybrid retrieval), Google Gemini API, SQLite và Streamlit.

## Quyết định thiết kế

- **Metadata `source_file` dùng tên file gốc.** Streamlit ghi file upload vào file tạm có tên ngẫu nhiên; tên file gốc được lưu cho mọi chunk để metadata filtering hoạt động ổn định.
- **Lịch sử chat và vector tách riêng.** SQLite giữ conversation; ChromaDB giữ kiến thức PDF. Xóa conversation không làm mất dữ liệu đã ingest.
- **Thiết kế pipeline theo từng lớp.** Ingest, retrieval, quản lý conversation và giao diện được tách riêng để có thể mở rộng về sau.

## Cài đặt

Yêu cầu: Python 3.10+ và [Gemini API key](https://aistudio.google.com/apikey).

```powershell
git clone https://github.com/huyvii/qa-agent.git
cd qa-agent
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Mở `.env` và điền:

```text
GOOGLE_API_KEY=your_key_here
```

Không commit `.env`, `chat_history.db` hoặc `chroma_db/`.

## Chạy

```powershell
streamlit run app.py
```

Upload PDF ở sidebar, tạo cuộc trò chuyện mới và đặt câu hỏi. Lần chạy đầu có thể chậm hơn do embedding model được nạp vào bộ nhớ.

## Đánh giá retrieval

`evaluation/retrieval_baseline.py` đo Hit@3 trên 10 câu hỏi tiếng Việt với evidence định sẵn trong CV mẫu; kết quả từng lần chạy lưu trong `evaluation/results/`.

| Cấu hình | Hit@3 |
|---|---|
| Vector all-MiniLM-L6-v2 (English-only) | 8/10 |
| Vector multilingual thuần | 6/10 |
| Hybrid: multilingual vector + BM25, hợp nhất RRF | 8/10 |

Điểm thú vị: multilingual thuần đỗ 2 case luận văn mà model English trượt (đúng nghĩa Việt↔Anh) nhưng lại trượt các case định danh (tên, GPA) cần khớp từ khóa chính xác. Hai nhánh gãy ở chỗ khác nhau — hybrid gom được ưu điểm cả hai. 2 case còn fail do chunk không chứa từ khóa nào của câu hỏi tiếng Việt; hướng giải quyết là query translation hoặc reranker đa ngữ.

## Cho người khác dùng (deploy)

Hai cách mở project thành link web cho người khác:

- **Streamlit Community Cloud (miễn phí, ~15 phút):** push code lên GitHub, kết nối repo tại share.streamlit.io, dán `GOOGLE_API_KEY` vào phần Secrets. Lưu ý: filesystem tạm thời — PDF đã ingest và lịch sử chat mất khi app restart, và mọi người dùng chung một kho tài liệu. Đủ cho demo/portfolio.
- **VPS hoặc Railway/Render + Docker (chi phí nhỏ):** mount volume cho `chroma_db/` và `chat_history.db` để dữ liệu bền, chủ động kiểm soát tài nguyên. Đủ cho người dùng thật dùng thử.

## Giới hạn

- Pipeline hiện tập trung vào text; bảng phức tạp, hình ảnh, công thức hoặc layout đặc biệt có thể trích xuất không chính xác. Trang scan được cảnh báo khi ingest; OCR chỉ nên thêm khi có tài liệu thực tế cần nó.
- ChromaDB và SQLite chạy local (đổi vị trí qua `CHROMA_DIR` / `CHAT_DB_PATH` để mount volume khi deploy). Mật khẩu gate là lớp chặn kiểu demo — production cần đăng nhập thật theo user.
- Xử lý lỗi chưa phủ hết các trường hợp PDF rỗng, file quá lớn hoặc lỗi API.

## Kiểm thử

```powershell
pip install -r requirements-dev.txt
python -m pytest tests -q
```

Bộ test phủ logic thuần (tokenize, RRF fusion, chuẩn hóa text), SQLite (roundtrip, cascade xóa, format timestamp) và hành vi hybrid retrieval qua fake vectorstore.

## Hướng phát triển

- Query translation hoặc reranker đa ngữ (bge-reranker) cho 2 case retrieval còn fail; mở rộng bộ eval ra nhiều loại PDF.
- Thêm Parent Document Retriever: search bằng chunk nhỏ nhưng gửi context lớn hơn cho Gemini.
- Bổ sung xử lý lỗi và unit test.
- Xây dựng bộ đánh giá trên nhiều loại tài liệu PDF.
- Chỉ thêm OCR/vision nếu xuất hiện PDF có bảng hoặc hình ảnh mà text extraction không xử lý tốt.

## Nếu deploy production thì cần gì?

Project hiện tại là prototype học tập. Để deploy production cần cân nhắc:

- **Multi-tenant isolation:** phân tách document, vector và conversation theo user/tenant.
- **Managed database:** dùng Supabase/Neon cho metadata và lịch sử; dùng vector database managed cho embeddings.
- **Object storage:** lưu PDF trong object storage thay vì filesystem local.
- **Secrets management:** dùng secret manager hoặc platform secrets, không commit API key.
- **Rate limiting và authentication:** giới hạn request, dung lượng upload và thời gian xử lý theo user.
- **Observability:** logging, latency tracking, retrieval evaluation và cảnh báo lỗi API.
