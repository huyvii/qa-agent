# QA Agent — RAG-based PDF Question Answering

QA Agent là ứng dụng hỏi đáp tài liệu PDF được xây dựng bằng kiến trúc **Retrieval-Augmented Generation (RAG)**. Người dùng upload một hoặc nhiều tài liệu, tạo các cuộc trò chuyện riêng và đặt câu hỏi dựa trên nội dung PDF.

Project tập trung vào ba vấn đề thực tế của một hệ thống RAG:

- Truy xuất đúng đoạn tài liệu liên quan trước khi gọi LLM.
- Phân tách dữ liệu giữa nhiều PDF và nhiều cuộc trò chuyện.
- Đo chất lượng retrieval bằng một bộ câu hỏi cố định thay vì chỉ đánh giá cảm tính.

## Kiến trúc

```text
PDF upload
    ↓
PyPDFLoader → RecursiveCharacterTextSplitter
    ↓
Local embeddings (all-MiniLM-L6-v2)
    ↓
ChromaDB + metadata source_file
    ↓
Similarity search theo từng tài liệu
    ↓
Prompt có context → Google Gemini
    ↓
Câu trả lời + nguồn tham khảo
```

SQLite được dùng riêng để quản lý lịch sử:

```text
SQLite
├── documents       — tài liệu đã ingest
├── conversations   — mỗi cuộc trò chuyện gắn với một source_file
└── messages        — lịch sử câu hỏi, câu trả lời và nguồn
```

## Tính năng hiện tại

- Upload và ingest tài liệu PDF.
- Chia tài liệu thành các chunk và lưu embedding local bằng `all-MiniLM-L6-v2`.
- Tạo, chuyển đổi và xóa nhiều cuộc trò chuyện.
- Lọc retrieval theo `source_file`, tránh trả lời lẫn dữ liệu giữa nhiều PDF.
- Xóa conversation chỉ xóa lịch sử trong SQLite; dữ liệu PDF trong Chroma vẫn được giữ lại.
- Hiển thị nguồn tham khảo theo tên file và số trang.
- Prompt ưu tiên trả lời dựa trên PDF; câu hỏi ngoài phạm vi có thể được trả lời bằng kiến thức chung nhưng phải được đánh dấu rõ là không dựa trên tài liệu.

## Công nghệ sử dụng

- **Python** — ngôn ngữ chính.
- **LangChain** — xây dựng pipeline loading, splitting, retrieval và prompting.
- **ChromaDB** — vector database chạy local.
- **Hugging Face sentence-transformers** — embedding model chạy trên CPU.
- **Google Gemini API** — sinh câu trả lời dựa trên context.
- **SQLite** — quản lý conversation và message history.
- **Streamlit** — giao diện web.

## Cài đặt

```powershell
git clone <repo-url>
cd qa-agent
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tạo file `.env` ở thư mục gốc và thêm Gemini API key:

```text
GOOGLE_API_KEY=your_key_here
```

Không commit `.env`, `chat_history.db` hoặc `chroma_db/` vào repository.

## Chạy ứng dụng

```powershell
streamlit run app.py
```

Sau đó upload file PDF trong sidebar, chọn **Nạp tài liệu / Tạo cuộc trò chuyện mới**, rồi đặt câu hỏi trong cuộc trò chuyện tương ứng.

Các script `ingest.py` và `query.py` là những script thử nghiệm CLI ban đầu. Luồng chính của project hiện nằm trong `app.py`, `rag_core.py` và `chat_db.py`.

## Đánh giá retrieval

Project có 10 câu hỏi cố định dựa trên CV mẫu. Mỗi câu được tính là một **hit** nếu một trong top 3 chunk retrieval chứa đoạn bằng chứng đã biết trước.

Baseline hiện tại:

```text
Metric:  Retrieval Hit@3
Result:  8/10 = 80%
Model:   sentence-transformers/all-MiniLM-L6-v2
```

Chạy lại baseline:

```powershell
python evaluation/retrieval_baseline.py --source-file "Hoang-An_Pham-AI-Engineer (2).pdf"
```

Kết quả chi tiết được lưu tại `evaluation/results/retrieval_baseline.json`. Đây là benchmark nội bộ trên một PDF mẫu, không đại diện cho mọi loại tài liệu PDF.

## Những gì đã học được

- RAG không chỉ là gọi LLM; chất lượng câu trả lời phụ thuộc trực tiếp vào đoạn context được retrieval.
- Metadata phải dùng một định danh tài liệu ổn định. File upload tạm có tên ngẫu nhiên sẽ làm filter multi-document bị sai.
- Conversation history và vector knowledge nên được quản lý độc lập: xóa một cuộc trò chuyện không nên xóa kiến thức PDF.
- Prompt cần phân biệt câu hỏi có trong tài liệu và câu hỏi ngoài phạm vi, thay vì từ chối mọi câu hỏi không khớp context.
- Cần có bộ test cố định và số liệu baseline trước khi thêm reranking hoặc thay đổi retriever.
- Embedding local giúp giảm chi phí API nhưng lần khởi tạo model đầu tiên có thể chậm và tốn RAM/CPU.

## Giới hạn hiện tại

- ChromaDB đang lưu local, phù hợp cho project học tập, demo.
- PDF được xử lý chủ yếu dưới dạng text; bảng phức tạp, hình ảnh hoặc layout đặc biệt có thể cần pipeline riêng.
- Retrieval baseline hiện mới đo trên một CV mẫu và 10 câu hỏi.
- Reranking, Parent Document Retriever, xử lý lỗi đầy đủ và unit test đang là các bước tiếp theo.

## Hướng phát triển

- Thêm BGE multilingual reranker và so sánh với baseline bằng cùng bộ test.
- Thêm Parent Document Retriever: search bằng chunk nhỏ nhưng gửi context lớn hơn cho Gemini.
- Bổ sung xử lý lỗi cho file sai định dạng, PDF rỗng, file quá lớn và lỗi API.
- Viết unit test cho các hàm ingest, retrieval và nguồn tham khảo.
- Giới hạn dung lượng upload cho môi trường local.
- Chỉ thêm pipeline OCR/vision nếu xuất hiện PDF có bảng hoặc hình ảnh mà text extraction không xử lý tốt.

## Nếu deploy production thì cần gì?

Project hiện tại là một prototype học tập. Để tiến tới production cần cân nhắc:

- **Multi-tenant isolation:** phân tách document, vector và conversation theo user/tenant.
- **Managed database:** dùng Supabase/Neon cho metadata và lịch sử; Qdrant Cloud/Chroma Cloud hoặc vector database managed cho embeddings.
- **Object storage:** lưu PDF trong S3-compatible storage thay vì filesystem local.
- **Secrets management:** dùng secret manager hoặc environment secrets, không commit API key.
- **Rate limiting và authentication:** giới hạn số request, dung lượng upload và thời gian xử lý mỗi user.
- **Observability:** logging, latency tracking, retrieval evaluation và cảnh báo lỗi API.
