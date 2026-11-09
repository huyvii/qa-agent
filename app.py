import hashlib
import hmac
import os
import tempfile
import time
import streamlit as st
from rag_core import (
    ingest_pdf,
    answer_stream,
    filter_sources_for_answer,
    get_vectorstore,
    delete_document_vectors,
    get_reranker,
    create_llm,
)
import chat_db

st.set_page_config(page_title="QA Agent", page_icon="📄", layout="wide")

chat_db.init_db()


# Gate mật khẩu tùy chọn: đặt APP_PASSWORD trong .env khi đưa link cho người khác.
# Đây là gate kiểu demo — production cần đăng nhập thật theo user (xem README).
def get_config_value(name: str) -> str | None:
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value).strip() if value else None


APP_PASSWORD = get_config_value("APP_PASSWORD")
GOOGLE_API_KEY = get_config_value("GOOGLE_API_KEY")

if APP_PASSWORD and not st.session_state.get("authenticated"):
    st.title("🔒 QA Agent")
    pw = st.text_input("Nhập mật khẩu để tiếp tục:", type="password")
    if st.button("Đăng nhập"):
        if hmac.compare_digest(pw, APP_PASSWORD):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Sai mật khẩu.")
    st.stop()

st.title("📄 QA Agent — Hỏi đáp tài liệu PDF")


@st.cache_resource
def load_vectorstore():
    return get_vectorstore()


@st.cache_resource
def load_llm():
    retries_value = get_config_value("GEMINI_MAX_RETRIES")
    try:
        retries = int(retries_value) if retries_value else None
    except ValueError as exc:
        raise RuntimeError("GEMINI_MAX_RETRIES phải là số nguyên không âm.") from exc
    return create_llm(
        api_key=GOOGLE_API_KEY,
        model_name=get_config_value("GEMINI_MODEL"),
        max_retries=retries,
    )


def request_error_message(error: Exception) -> str:
    if isinstance(error, RuntimeError):
        return str(error)
    message = str(error).casefold()
    status = str(
        getattr(error, "status_code", None) or getattr(error, "code", "")
    ).casefold()
    if status == "429" or "resource_exhausted" in message or "quota" in message:
        return (
            "Gemini đang giới hạn quota/tốc độ cho project này (HTTP 429). "
            "Chờ quota hồi lại hoặc kiểm tra mục Rate limits trong Google AI Studio."
        )
    if status in {"401", "403"} or "api key not valid" in message:
        return (
            "Gemini từ chối xác thực. Kiểm tra GOOGLE_API_KEY trong .env hoặc Streamlit Secrets, "
            "và xác nhận key thuộc project đang bật Gemini API."
        )
    if status == "404" or "model not found" in message:
        return "Không tìm thấy model Gemini. Kiểm tra giá trị GEMINI_MODEL trong cấu hình."
    return f"Yêu cầu thất bại ({type(error).__name__}). Kiểm tra log của Streamlit để xem chi tiết."


def _format_conversation_title(question: str, max_chars: int = 42) -> str:
    cleaned = " ".join((question or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars - 1].rstrip()}…"


if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None
if "pending_delete_document" not in st.session_state:
    st.session_state.pending_delete_document = None


# ==================== Sidebar ====================
with st.sidebar:
    st.header("📂 Tải tài liệu")
    uploaded_file = st.file_uploader("Chọn file PDF", type="pdf", max_upload_size=30)

    if not GOOGLE_API_KEY:
        st.warning("Chưa có GOOGLE_API_KEY. Thêm key vào .env hoặc Streamlit Secrets để hỏi đáp.")

    if uploaded_file is not None:
        if st.button("Nạp tài liệu / Tạo cuộc trò chuyện mới"):
            file_name = uploaded_file.name
            content_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            existing = chat_db.get_document(file_name)

            if existing and existing["content_hash"] == content_hash:
                st.info(f"'{file_name}' đã được nạp trước đó với cùng nội dung — dùng lại dữ liệu cũ, không nhúng lại.")
            else:
                with st.spinner("Đang xử lý tài liệu..."):
                    ingest_timings = {}
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        result = ingest_pdf(
                            tmp_path,
                            source_file=file_name,
                            timings=ingest_timings,
                        )
                    except Exception as exc:
                        if existing and existing.get("content_hash"):
                            try:
                                delete_document_vectors(
                                    file_name,
                                    keep_content_hash=existing["content_hash"],
                                )
                            except Exception:
                                pass
                        st.error(
                            f"Không nạp được PDF ({type(exc).__name__}). "
                            "Kiểm tra file có hợp lệ và có thể trích xuất văn bản không."
                        )
                        st.stop()
                    finally:
                        os.unlink(tmp_path)

                    try:
                        # Chỉ dọn phiên bản cũ sau khi bản mới đã được lập chỉ mục thành công.
                        delete_document_vectors(file_name, keep_content_hash=content_hash)
                    except Exception:
                        st.warning(
                            "PDF mới đã được nạp, nhưng chưa dọn được các vector phiên bản cũ. "
                            "Kiểm tra kho Chroma trước khi tiếp tục hỏi đáp file này."
                        )
                    chat_db.save_document(file_name, content_hash, result["chunks"])
                    # Reopen the persistent collection after adding new chunks.
                    load_vectorstore.clear()

                st.success(f"Đã nạp xong! ({result['chunks']} đoạn văn bản)")
                if result["empty_pages"]:
                    st.warning(
                        f"{len(result['empty_pages'])} trang gần như không có text "
                        f"(trang {[p + 1 for p in result['empty_pages']]}) — có thể là bản "
                        "scan/hình ảnh, trả lời sẽ kém trên các trang này."
                    )
                st.caption(
                    "⏱ Ingest: "
                    f"đọc {ingest_timings['loader_seconds']:.2f}s · "
                    f"chia đoạn {ingest_timings['splitter_seconds']:.2f}s · "
                    f"load model {ingest_timings['embedding_model_load_seconds']:.2f}s · "
                    f"embedding + lưu Chroma {ingest_timings['embedding_and_index_seconds']:.2f}s · "
                    f"tổng {ingest_timings['total_seconds']:.2f}s"
                )

            new_conv_id = chat_db.create_conversation(source_file=file_name)
            st.session_state.active_conversation_id = new_conv_id
            st.rerun()

    st.divider()
    st.subheader("Tài liệu đã nạp")
    documents = chat_db.get_documents()
    if not documents:
        st.caption("Chưa có tài liệu nào trong kho.")
    for document in documents:
        source_file = document["source_file"]
        info_col, chat_col, action_col = st.columns([4, 1, 1])
        info_col.caption(f"{source_file} · {document['chunk_count']} đoạn")
        if chat_col.button(
            "➕",
            key=f"new_chat_{source_file}",
            help=f"Tạo cuộc trò chuyện mới với {source_file}",
        ):
            new_conv_id = chat_db.create_conversation(source_file=source_file)
            st.session_state.active_conversation_id = new_conv_id
            st.rerun()
        if action_col.button(
            "🗑️",
            key=f"delete_doc_{source_file}",
            help=f"Xóa tài liệu {source_file}",
        ):
            st.session_state.pending_delete_document = source_file
            st.rerun()

        if st.session_state.pending_delete_document == source_file:
            st.warning(
                f"Xóa '{source_file}' sẽ xóa vector đã nạp cùng toàn bộ cuộc trò chuyện "
                "và lịch sử gắn với file này. File PDF gốc trên máy bạn không bị xóa."
            )
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("Xác nhận xóa", key=f"confirm_doc_{source_file}"):
                try:
                    delete_document_vectors(source_file)
                    chat_db.delete_document(source_file)
                    active_id = st.session_state.active_conversation_id
                    active_conversation = chat_db.get_conversation(active_id) if active_id else None
                    if active_conversation and active_conversation["source_file"] == source_file:
                        st.session_state.active_conversation_id = None
                    st.session_state.pending_delete_document = None
                    load_vectorstore.clear()
                except Exception:
                    st.error("Không xóa được tài liệu. Kiểm tra kho Chroma và SQLite rồi thử lại.")
                else:
                    st.rerun()
            if cancel_col.button("Hủy", key=f"cancel_doc_{source_file}"):
                st.session_state.pending_delete_document = None
                st.rerun()

    st.divider()
    st.header("💬 Cuộc trò chuyện")

    conversations = chat_db.get_conversations()
    if not conversations:
        st.caption("Chưa có cuộc trò chuyện nào. Tải 1 file PDF để bắt đầu.")

    for conv in conversations:
        col1, col2 = st.columns([5, 1])
        is_active = conv["id"] == st.session_state.active_conversation_id

        with col1:
            label = f"{'🟢 ' if is_active else ''}{conv['title']}"
            if st.button(
                label,
                key=f"conv_{conv['id']}",
                help=f"Tài liệu: {conv['source_file']}",
                use_container_width=True,
            ):
                st.session_state.active_conversation_id = conv["id"]
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"del_{conv['id']}", help="Xóa cuộc trò chuyện"):
                chat_db.delete_conversation(conv["id"])
                if is_active:
                    st.session_state.active_conversation_id = None
                st.rerun()


# ==================== Khung chat chính ====================
active_id = st.session_state.active_conversation_id
if active_id is not None and chat_db.get_conversation(active_id) is None:
    active_id = None
    st.session_state.active_conversation_id = None

if active_id is None:
    st.info("👈 Tải 1 file PDF hoặc chọn 1 cuộc trò chuyện ở sidebar để bắt đầu.")
else:
    active_conv = chat_db.get_conversation(active_id)
    st.caption(f"📄 Tài liệu: {active_conv['source_file']}")

    messages = chat_db.get_messages(active_id)
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["sources"]:
                with st.expander("Nguồn tham khảo"):
                    for s in msg["sources"]:
                        st.write(f"- {s}")

    question = st.chat_input("Nhập câu hỏi về tài liệu...")

    if question:
        if not GOOGLE_API_KEY:
            st.error("Chưa cấu hình GOOGLE_API_KEY. Thêm key vào .env hoặc Streamlit Secrets rồi thử lại.")
            st.stop()
        # Lịch sử đầy đủ nằm trong SQLite; rag_core chỉ gửi vài lượt phù hợp cho Gemini.
        history = [{"role": m["role"], "content": m["content"]} for m in messages]

        with st.chat_message("user"):
            st.write(question)

        try:
            with st.chat_message("assistant"):
                request_started = time.perf_counter()
                answer_timings = {}
                with st.spinner("Đang tìm câu trả lời..."):
                    resource_started = time.perf_counter()
                    vectorstore = load_vectorstore()
                    llm = load_llm()
                    reranker = get_reranker()
                    resource_load_seconds = time.perf_counter() - resource_started
                    answer_gen, sources = answer_stream(
                        question,
                        vectorstore,
                        llm,
                        source_file_filter=active_conv["source_file"],
                        history=history,
                        timings=answer_timings,
                        reranker=reranker,
                    )
                # stream từng token; write_stream trả về text đầy đủ để lưu DB
                full_answer = st.write_stream(answer_gen)
                shown_sources = filter_sources_for_answer(full_answer, sources)
                memory_seconds = answer_timings.get("memory_seconds", 0.0)
                st.caption(
                    "⏱ "
                    f"load resource {resource_load_seconds:.2f}s · "
                    f"ghi nhớ cục bộ {memory_seconds:.2f}s · "
                    f"retrieval {answer_timings['retrieval_seconds']:.2f}s · "
                    f"rerank {answer_timings.get('rerank_seconds', 0.0):.2f}s · "
                    f"Gemini {answer_timings['llm_seconds']:.2f}s · "
                    f"tổng {time.perf_counter() - request_started:.2f}s"
                )
                if shown_sources:
                    with st.expander("Nguồn tham khảo"):
                        for s in shown_sources:
                            st.write(f"- {s}")

            # Chỉ lưu cặp câu hỏi - trả lời khi yêu cầu thành công để tránh tin nhắn lỗi mồ côi.
            chat_db.save_message(active_id, "user", question)
            chat_db.save_message(active_id, "assistant", full_answer, shown_sources)
            if not messages and active_conv["title"] == active_conv["source_file"]:
                chat_db.update_conversation_title(
                    active_id, _format_conversation_title(question)
                )
        except Exception as exc:
            with st.chat_message("assistant"):
                st.error(request_error_message(exc))
