# QA Agent — RAG-based Document Question Answering

Hệ thống hỏi đáp dựa trên tài liệu PDF, sử dụng kỹ thuật RAG (Retrieval-Augmented Generation). Người dùng upload file PDF, hệ thống trả lời câu hỏi dựa trên nội dung tài liệu, có trích dẫn nguồn (trang cụ thể).

## Kiến trúc
PDF → Text Splitter (chunking) → Embedding (sentence-transformers, local)
→ Chroma Vector DB → Semantic Search → Gemini 3.6 Flash → Câu trả lời
## Công nghệ sử dụng

- **LangChain** — orchestrate pipeline RAG
- **ChromaDB** — vector database, lưu local
- **sentence-transformers (all-MiniLM-L6-v2)** — embedding model, chạy CPU, không cần GPU
- **Google Gemini API (gemini-3.6-flash)** — sinh câu trả lời
- **Streamlit** — giao diện web

## Cài đặt

```bash
git clone <repo-url>
cd qa-agent
python -m venv venv
venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

Tạo file `.env` từ mẫu `.env.example`, điền Gemini API key (lấy tại https://aistudio.google.com/apikey):

```text
GOOGLE_API_KEY=your_key_here
```

## Sử dụng

Ingest tài liệu vào vector DB:
```bash
python ingest.py
```

Chạy hỏi đáp (CLI):
```bash
python query.py
```

Chạy giao diện web:
```bash
streamlit run app.py
```

## Những gì đã học được / thử thách kỹ thuật

- Kiểm soát hallucination bằng prompt engineering (ràng buộc model chỉ trả lời dựa trên context)
- Đánh đổi giữa chunk size và chất lượng câu trả lời / token cost
- Semantic search với embedding model chạy local, không phụ thuộc GPU

## Hướng phát triển tiếp theo

- Reranking kết quả search (BGE-Reranker)
- Parent Document Retriever để giữ ngữ cảnh tốt hơn
- Hỗ trợ đa tài liệu với metadata filtering