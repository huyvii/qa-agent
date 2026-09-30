import os
import time
from functools import lru_cache
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
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


def answer_question(
    question: str,
    vectorstore,
    llm,
    source_file_filter: str = None,
    k: int = 3,
    timings: dict | None = None,
):
    """Search + gọi Gemini, trả về (câu trả lời, danh sách nguồn).

    source_file_filter: nếu truyền vào, chỉ search trong phạm vi file đó
    (dùng metadata filtering của Chroma) — đây là cách fix bug trộn dữ liệu
    giữa nhiều file khi có nhiều conversation cùng tồn tại.
    """
    search_kwargs = {"k": k}
    if source_file_filter:
        search_kwargs["filter"] = {"source_file": source_file_filter}

    question_started = time.perf_counter()

    retrieval_started = time.perf_counter()
    results = vectorstore.similarity_search(question, **search_kwargs)
    retrieval_seconds = time.perf_counter() - retrieval_started
    context = "\n\n".join([doc.page_content for doc in results])

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
    llm_started = time.perf_counter()
    response = chain.invoke({"context": context, "question": question})
    llm_seconds = time.perf_counter() - llm_started

    # thay vì string thuần -> phải chuẩn hóa lại thành text trước khi dùng/lưu DB
    if isinstance(response.content, str):
        answer_text = response.content
    else:
        answer_text = "".join(
            block.get("text", "") for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    sources = [
        f"{doc.metadata.get('source_file', '?')} - trang {doc.metadata.get('page', '?')}"
        for doc in results
    ]

    if timings is not None:
        timings.update({
            "retrieval_seconds": retrieval_seconds,
            "llm_seconds": llm_seconds,
            "total_seconds": time.perf_counter() - question_started,
            "retrieved_chunks": len(results),
        })

    return answer_text, sources
