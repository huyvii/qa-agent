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
- Không ingest lại tài liệu đã có cùng tên file.
- Xóa cuộc trò chuyện chỉ xóa lịch sử chat, không xóa vector của PDF.
- Ưu tiên trả lời theo PDF; với câu hỏi ngoài phạm vi tài liệu, câu trả lời được đánh dấu rõ là kiến thức chung.

## Cách hoạt động

```text
PDF → tách chunk → embedding local (all-MiniLM-L6-v2)
    → lưu vào ChromaDB kèm metadata (source_file, page)
    → semantic search top-k, lọc theo source_file
    → ghép context vào prompt → Gemini sinh câu trả lời
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

**Công nghệ:** Python, LangChain, ChromaDB, sentence-transformers, Google Gemini API, SQLite và Streamlit.

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

## Giới hạn

- Pipeline hiện tập trung vào text; bảng phức tạp, hình ảnh, công thức hoặc layout đặc biệt có thể trích xuất không chính xác.
- ChromaDB và SQLite chạy local, phù hợp học tập/demo; chưa có đăng nhập hoặc phân quyền người dùng.
- File được nhận diện theo tên. Hai PDF khác nội dung nhưng trùng tên hiện có thể bị xem là cùng tài liệu.
- Chưa có unit test và xử lý lỗi đầy đủ cho PDF rỗng, file quá lớn hoặc lỗi API.

## Hướng phát triển

- Thêm BGE multilingual reranker và đánh giá chất lượng retrieval sau khi pipeline ổn định.
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
