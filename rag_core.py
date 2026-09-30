import os
import re
import time
from functools import lru_cache
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

load_dotenv()

# Embedding multilingual — cần cho truy vấn tiếng Việt trên tài liệu tiếng Anh.
# Đổi model là phải re-ingest toàn bộ: vector cũ lệch không gian với model mới.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
CHROMA_DIR = "./chroma_db"


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the local embedding model once per application process."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def ingest_pdf(
    file_path: str,
    source_file: str = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    timings: dict | None = None,
):
    """Đọc 1 file PDF, chia nhỏ, lưu vào Chroma. Trả về số chunk đã lưu.
    Lưu ý: hàm này KHÔNG tự kiểm tra trùng lặp — việc kiểm tra file đã ingest
    hay chưa (qua chat_db.document_exists) phải làm ở tầng gọi (app.py)."""
    ingest_started = time.perf_counter()

    loader_started = time.perf_counter()
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    loader_seconds = time.perf_counter() - loader_started

    # file_path may be a temporary upload path.  The stable original filename is
    # what conversations use to filter Chroma, so it must be stored in metadata.
    file_name = source_file or os.path.basename(file_path)
    for doc in documents:
        doc.metadata["source_file"] = file_name

    splitter_started = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    splitter_seconds = time.perf_counter() - splitter_started

    model_started = time.perf_counter()
    embedding_model = get_embedding_model()
    embedding_model_load_seconds = time.perf_counter() - model_started

    indexing_started = time.perf_counter()
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )
    embedding_and_index_seconds = time.perf_counter() - indexing_started

    if timings is not None:
        timings.update({
            "loader_seconds": loader_seconds,
            "splitter_seconds": splitter_seconds,
            "embedding_model_load_seconds": embedding_model_load_seconds,
            "embedding_and_index_seconds": embedding_and_index_seconds,
            "total_seconds": time.perf_counter() - ingest_started,
            "chunks": len(chunks),
        })

    return len(chunks)


def get_vectorstore():
    embedding_model = get_embedding_model()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )


# ==================== hỏi đáp (streaming + multi-turn) ====================

HISTORY_TURNS = 4          # số lượt chat gần nhất dùng khi viết lại câu hỏi
MAX_HISTORY_CHARS = 500    # giới hạn mỗi lượt để prompt không phình to

CONDENSE_PROMPT = """Bạn viết lại câu hỏi nối tiếp của người dùng thành MỘT câu hỏi ĐỘC LẬP,
có đủ ngữ cảnh để tra cứu tài liệu mà không cần biết các lượt chat trước.

- Chỉ xuất ra câu hỏi đã viết lại, không giải thích gì thêm.
- Giữ nguyên ngôn ngữ của câu hỏi gốc.
- Nếu câu hỏi gốc đã tự đủ ngữ cảnh, trả lại nguyên văn.

Lịch sử chat:
{history}

Câu hỏi nối tiếp: {question}

Câu hỏi độc lập:"""


def _extract_text(content) -> str:
    """Response của Gemini có thể là string hoặc list các block dict."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def condense_question(question: str, history: list | None, llm) -> str:
    """Dùng LLM biến câu hỏi nối tiếp (vd: "còn GPA thạc sĩ thì sao?") thành câu
    hỏi độc lập, để retrieval không bị sai khi chỉ dựa vào câu nối thiếu ngữ cảnh."""
    if not history:
        return question
    lines = []
    for msg in history[-2 * HISTORY_TURNS:]:
        role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {(msg['content'] or '')[:MAX_HISTORY_CHARS]}")
    prompt = ChatPromptTemplate.from_template(CONDENSE_PROMPT)
    response = (prompt | llm).invoke({"history": "\n".join(lines), "question": question})
    return _extract_text(response).strip() or question


RRF_K = 60          # hằng số của Reciprocal Rank Fusion
FETCH_MULT = 3      # mỗi nhánh lấy k*FETCH_MULT kết quả rồi mới fuse


def _tokenize(text: str) -> list[str]:
    """Tách từ cho BM25, hỗ trợ unicode (tiếng Việt có dấu, không phân tách kiểu tiếng Anh)."""
    return re.findall(r"\w+", text.lower(), re.UNICODE)


def _rrf_fuse(list_a: list[Document], list_b: list[Document], k: int) -> list[Document]:
    """Hợp nhất 2 danh sách kết quả xếp hạng bằng Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranked in (list_a, list_b):
        for rank, doc in enumerate(ranked):
            key = doc.page_content
            docs[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ordered[:k]]


def retrieve(
    question: str,
    vectorstore,
    source_file_filter: str | None = None,
    k: int = 3,
    hybrid: bool = True,
):
    """Hybrid retrieval: semantic search (vector) + BM25 (lexical), hợp nhất bằng RRF.

    Hai nhánh gãy ở chỗ khác nhau: vector bắt được câu hỏi diễn đạt khác cách viết
    trong tài liệu nhưng hay trượt từ khóa chính xác (số liệu, tên riêng); BM25
    ngược lại. Eval thực nghiệm: vector EN 8/10, vector multilingual 6/10, hybrid cao hơn cả hai.
    """
    where = {"source_file": source_file_filter} if source_file_filter else None
    vec_docs = vectorstore.similarity_search(question, k=k * FETCH_MULT, filter=where)
    if not hybrid:
        return vec_docs[:k]

    # BM25 chạy trên toàn bộ chunk của file — với prototype (vài trăm chunk) là đủ;
    # corpus lớn thì cần đưa BM25 sang index riêng (Elasticsearch/SQLite FTS).
    data = vectorstore.get(where=where or {}, include=["documents", "metadatas"])
    texts = data.get("documents") or []
    if not texts:
        return vec_docs[:k]

    corpus = [
        Document(page_content=t or "", metadata=m or {})
        for t, m in zip(texts, data.get("metadatas") or [])
    ]
    bm25 = BM25Okapi([_tokenize(d.page_content) for d in corpus])
    bm25_scores = bm25.get_scores(_tokenize(question))
    ranked_idx = sorted(range(len(corpus)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_docs = [corpus[i] for i in ranked_idx[: k * FETCH_MULT]]

    return _rrf_fuse(vec_docs, bm25_docs, k)


def answer_stream(
    question: str,
    vectorstore,
    llm,
    source_file_filter: str = None,
    k: int = 3,
    history: list | None = None,
    timings: dict | None = None,
):
    """Chuẩn bị retrieval rồi trả về (token_generator, sources).

    Multi-turn: nếu có history, câu hỏi được viết lại cho độc lập trước khi
    retrieve; LLM trả lời dựa trên câu hỏi gốc kèm context đã retrieve.
    Generator stream từng token; caller phải tiêu thụ hết generator để
    timings có llm_seconds/total_seconds.
    """
    started = time.perf_counter()

    condense_started = time.perf_counter()
    standalone = condense_question(question, history, llm)
    if timings is not None:
        timings["condense_seconds"] = time.perf_counter() - condense_started
        timings["standalone_question"] = standalone

    retrieval_started = time.perf_counter()
    results = retrieve(standalone, vectorstore, source_file_filter, k)
    if timings is not None:
        timings["retrieval_seconds"] = time.perf_counter() - retrieval_started

    context = "\n\n".join([doc.page_content for doc in results])
    sources = [
        f"{doc.metadata.get('source_file', '?')} - trang {doc.metadata.get('page', '?')}"
        for doc in results
    ]

    prompt = ChatPromptTemplate.from_template(
        """Bạn là trợ lý AI hỏi đáp tài liệu PDF. Trả lời bằng tiếng Việt.

Ưu tiên trước hết: kiểm tra ngữ cảnh tài liệu bên dưới có liên quan đến câu hỏi hay không.

- Nếu ngữ cảnh có thông tin liên quan, trả lời dựa trên ngữ cảnh. Không thêm chi tiết
  mà tài liệu không nêu và không nói đó là kiến thức bên ngoài.
- Nếu câu hỏi nằm ngoài phạm vi tài liệu hoặc ngữ cảnh không đề cập đủ để trả lời,
  bạn vẫn có thể trả lời bằng kiến thức chung nếu phù hợp. Khi đó phải mở đầu rõ ràng
  bằng "Ngoài phạm vi tài liệu:" và nói rằng câu trả lời này không dựa trên PDF.
- Không bịa nội dung, số liệu, kinh nghiệm hoặc nguồn tham khảo cho tài liệu.
- Nếu không thể đưa ra một câu trả lời hữu ích ngay cả bằng kiến thức chung, hãy nói rõ
  điều đó thay vì đoán.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""
    )
    chain = prompt | llm

    def generate():
        llm_started = time.perf_counter()
        for chunk in chain.stream({"context": context, "question": question}):
            text = _extract_text(chunk.content)
            if text:
                yield text
        if timings is not None:
            timings["llm_seconds"] = time.perf_counter() - llm_started
            timings["total_seconds"] = time.perf_counter() - started

    return generate(), sources


def answer_question(
    question: str,
    vectorstore,
    llm,
    source_file_filter: str = None,
    k: int = 3,
    history: list | None = None,
    timings: dict | None = None,
):
    """Phiên bản không streaming của answer_stream — tiện cho test/eval."""
    generator, sources = answer_stream(
        question, vectorstore, llm, source_file_filter, k, history, timings
    )
    answer_text = "".join(generator)
    return answer_text, sources
