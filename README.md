# QA Agent

Ứng dụng hỏi đáp trên PDF bằng RAG: trích xuất văn bản, tìm đoạn liên quan, rồi dùng Gemini tạo câu trả lời có nguồn và số trang.

## Tính năng

- Tải lên nhiều PDF, tạo nhiều cuộc trò chuyện và lọc tìm kiếm theo đúng file.
- Tìm kiếm lai gồm embedding đa ngữ và BM25, hợp nhất bằng Reciprocal Rank Fusion (RRF), sau đó có thể xếp hạng lại bằng cross-encoder.
- Cache BM25 theo collection để tránh dựng lại index cho mỗi câu hỏi.
- Ghi nhớ toàn bộ lịch sử trong SQLite theo từng cuộc trò chuyện; dùng bốn lượt gần nhất và tối đa hai lượt cũ liên quan trong phạm vi 60 lượt trước đó.
- Stream câu trả lời; bổ sung ngữ cảnh cho câu hỏi nối tiếp bằng retrieval cục bộ, không thêm lượt gọi Gemini.
- Lưu lịch sử và thông tin file trong SQLite; lưu vector trên ChromaDB.
- Nhận diện file bằng SHA-256: bỏ qua nội dung không đổi; chỉ dọn vector cũ sau khi bản mới được lập chỉ mục thành công.
- Xóa tài liệu có bước xác nhận và xóa luôn các cuộc trò chuyện gắn với tài liệu đó.
- Cảnh báo trang PDF gần như không có văn bản và báo lỗi rõ khi không trích xuất được nội dung.
- Hỗ trợ key từ `.env` hoặc Streamlit Secrets; giới hạn file tải lên ở 30 MB.

## Luồng xử lý

```text
PDF → PyPDFLoader → chia đoạn → embedding đa ngữ → ChromaDB
                                           ↘ BM25 theo file
câu hỏi + bộ nhớ hội thoại → vector + BM25 → RRF → cross-encoder
                                            → Gemini → câu trả lời + nguồn
```

Khác với hỏi một AI tổng quát, project tìm bằng chứng trong đúng PDF đã tải lên, lọc theo file và trả nguồn theo trang. Embedding, BM25, reranker và truy hồi lịch sử chạy cục bộ. Gemini nhận câu hỏi, tối đa 3 đoạn PDF liên quan, bốn lượt gần nhất và tối đa hai lượt cũ được chọn theo mức liên quan trong 60 lượt trước đó. Toàn bộ PDF và toàn bộ lịch sử không được gửi lên.

| Tệp | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit, cấu hình và xử lý lỗi thân thiện |
| `rag_core.py` | Nạp PDF, embedding, retrieval và gọi Gemini |
| `chat_db.py` | Lịch sử hội thoại và danh sách PDF trong SQLite |
| `ingest.py` | Lệnh nạp PDF từ terminal |
| `query.py` | Lệnh hỏi đáp từ terminal |
| `evaluation/retrieval_baseline.py` | Đánh giá Hit@k trên bộ bài toán mẫu |
| `tests/` | Kiểm thử logic retrieval và SQLite |

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

Mở `.env`, thay giá trị mẫu bằng key của bạn:

```text
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
GEMINI_MAX_RETRIES=2
```

`GEMINI_MODEL` và `GEMINI_MAX_RETRIES` là tùy chọn. `GOOGLE_API_KEY` không được commit. Khi deploy Streamlit Community Cloud, đặt key trong Secrets; app đọc được cả Secrets lẫn biến môi trường.

## Chạy ứng dụng

```powershell
streamlit run app.py
```

Tải PDF ở sidebar, tạo cuộc trò chuyện rồi đặt câu hỏi. Lần chạy đầu có thể mất thời gian để tải embedding và reranker. App chỉ đọc PDF gốc qua file tạm trong lúc ingest; sau đó giữ vector và metadata, không giữ bản PDF gốc. `.env`, `chroma_db/` và `chat_history.db` đã được bỏ qua trong `.gitignore`.

Có thể dùng terminal thay giao diện:

```powershell
python ingest.py .\path\to\document.pdf
python query.py "Tóm tắt nội dung chính" --source-file document.pdf
```

`--source-file` phải giống tên file được dùng khi nạp để lọc đúng tài liệu.

## Giới hạn tốc độ Gemini

Ứng dụng **không tự đặt giới hạn số câu hỏi**. Gemini API áp quota theo project và model; giới hạn có thể phụ thuộc tier, số request mỗi phút, lượng token mỗi phút và số request mỗi ngày. Khi vượt quota, Google có thể trả HTTP 429. Xem [quota hiện hành](https://ai.google.dev/gemini-api/docs/rate-limits) trong Google AI Studio vì giới hạn thay đổi theo project và thời điểm.

Mỗi câu hỏi thường dùng một lượt gọi Gemini, kể cả câu hỏi nối tiếp; app xử lý bộ nhớ và bổ sung truy vấn bằng thuật toán cục bộ. App đặt tối đa 2 lần retry lỗi tạm thời; chỉnh bằng `GEMINI_MAX_RETRIES`. Retry không làm tăng quota khả dụng. Embedding và reranker chạy cục bộ, nên lần đầu có thể chậm và cần CPU/RAM.

Nếu gặp lỗi:

- **Thiếu key:** thêm `GOOGLE_API_KEY` vào `.env` hoặc Streamlit Secrets.
- **401/403:** kiểm tra key còn hiệu lực, đúng Google Cloud project và API được bật.
- **429:** đợi quota hồi lại hoặc xem tier/quota của project trong AI Studio.
- **Lỗi model:** kiểm tra `GEMINI_MODEL`; mặc định là `gemini-3.6-flash`.

## Đánh giá retrieval

```powershell
python evaluation/retrieval_baseline.py
python evaluation/retrieval_baseline.py --rerank
```

Mặc định, script lập chỉ mục bộ bài toán trong `tests/fixtures/math_exercises.md` vào Chroma tạm, đo Hit@3 bằng pipeline retrieval thật rồi xóa kho tạm. Script không gọi Gemini; embedding model có thể được tải ở lần đầu. Kết quả ghi vào `evaluation/results/retrieval_baseline.json`.

Muốn đo kho Chroma hiện có, truyền thư mục và tên file đã nạp:

```powershell
python evaluation/retrieval_baseline.py --chroma-dir .\chroma_db --source-file "math_exercises.pdf"
```

Các con số trong kết quả cũ đã bị gỡ vì chúng dựa trên một PDF cá nhân không có trong repo và không thể tái tạo từ project. Bộ hiện tại dùng bài toán mẫu; không so sánh trực tiếp với kết quả cũ.

## Kiểm thử

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

## Lưu trữ và bảo mật

- `CHROMA_DIR` đổi đường dẫn vector DB; `CHAT_DB_PATH` đổi đường dẫn SQLite.
- Mặc định, hai kho dữ liệu nằm cạnh source code nên app chạy ổn định kể cả khi khởi chạy từ thư mục khác.
- `APP_PASSWORD` bật một mật khẩu chung trước khi mở giao diện. Đây chỉ là cổng đơn giản cho demo, không phải xác thực riêng theo người dùng.
- Bộ nhớ chỉ chia theo conversation, không chia theo tài khoản. Mọi người mở cùng bản chạy hiện thấy chung danh sách chat và kho PDF; chưa có đăng nhập hay phân vùng tenant.
- Lịch sử hội thoại được giữ trên máy trong SQLite sau khi đóng/mở app. Không có bộ nhớ dùng chung giữa các conversation.
- ChromaDB và SQLite nằm trên đĩa local. Khi deploy cần volume bền vững; nếu nền tảng xóa filesystem tạm, vector và lịch sử sẽ mất.
- Pipeline hiện trích xuất text. PDF scan cần OCR; bảng phức tạp, công thức và bố cục nhiều cột có thể bị trích xuất sai.

## Hướng cải thiện tiếp theo

- Thêm xác thực và tách dữ liệu theo người dùng nếu mở cho nhiều người.
- Thêm backup/restore cho SQLite và Chroma, cùng thao tác xóa tài liệu rõ ràng.
- Đo latency, dung lượng model và chất lượng retrieval với PDF thật thuộc nhiều lĩnh vực.
- Chỉ thêm OCR, xử lý bảng hoặc vector database managed khi nhu cầu thực tế yêu cầu.
