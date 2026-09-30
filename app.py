import hashlib
import hmac
import streamlit as st
import tempfile
import os
import time
from rag_core import ingest_pdf, answer_stream, get_vectorstore, delete_document_vectors
from langchain_google_genai import ChatGoogleGenerativeAI
import chat_db

st.set_page_config(page_title="QA Agent", page_icon="📄", layout="wide")

chat_db.init_db()

# Gate mật khẩu tùy chọn: đặt APP_PASSWORD trong .env khi đưa link cho người khác.
# Đây là gate kiểu demo — production cần đăng nhập thật theo user (xem README).
APP_PASSWORD = os.getenv("APP_PASSWORD")

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
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None


# ==================== Sidebar ====================
with st.sidebar:
    st.header("📂 Tải tài liệu")
    uploaded_file = st.file_uploader("Chọn file PDF", type="pdf")

    if uploaded_file is not None:
        if st.button("Nạp tài liệu / Tạo cuộc trò chuyện mới"):
            file_name = uploaded_file.name
            content_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            existing = chat_db.get_document(file_name)

            if existing and existing["content_hash"] == content_hash:
                st.info(f"'{file_name}' đã được nạp trước đó với cùng nội dung — dùng lại dữ liệu cũ, không nhúng lại.")
            else:
                if existing:
                    # Trùng tên nhưng khác nội dung: thay thế vector cũ bằng nội dung mới
                    delete_document_vectors(file_name)

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
                    finally:
                        os.unlink(tmp_path)

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
    st.header("💬 Cuộc trò chuyện")

    conversations = chat_db.get_conversations()
    if not conversations:
        st.caption("Chưa có cuộc trò chuyện nào. Tải 1 file PDF để bắt đầu.")

    for conv in conversations:
        col1, col2 = st.columns([5, 1])
        is_active = conv["id"] == st.session_state.active_conversation_id

        with col1:
            label = f"{'🟢 ' if is_active else ''}{conv['title']}"
            if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                st.session_state.active_conversation_id = conv["id"]
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"del_{conv['id']}"):
                chat_db.delete_conversation(conv["id"])
                if is_active:
                    st.session_state.active_conversation_id = None
                st.rerun()


# ==================== Khung chat chính ====================
active_id = st.session_state.active_conversation_id

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
        # messages được nạp trước khi lưu câu hỏi mới nên history chưa gồm câu này
        history = [{"role": m["role"], "content": m["content"]} for m in messages[-8:]]

        chat_db.save_message(active_id, "user", question)
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            request_started = time.perf_counter()
            resource_started = time.perf_counter()
            vectorstore = load_vectorstore()
            llm = load_llm()
            resource_load_seconds = time.perf_counter() - resource_started
            answer_timings = {}
            with st.spinner("Đang tìm câu trả lời..."):
                answer_gen, sources = answer_stream(
                    question, vectorstore, llm,
                    source_file_filter=active_conv["source_file"],
                    history=history,
                    timings=answer_timings,
                )
            # stream từng token; write_stream trả về text đầy đủ để lưu DB
            full_answer = st.write_stream(answer_gen)
            condense_seconds = answer_timings.get("condense_seconds", 0.0)
            st.caption(
                "⏱ "
                f"load resource {resource_load_seconds:.2f}s · "
                f"viết lại câu hỏi {condense_seconds:.2f}s · "
                f"retrieval {answer_timings['retrieval_seconds']:.2f}s · "
                f"Gemini {answer_timings['llm_seconds']:.2f}s · "
                f"tổng {time.perf_counter() - request_started:.2f}s"
            )
            with st.expander("Nguồn tham khảo"):
                for s in sources:
                    st.write(f"- {s}")

        chat_db.save_message(active_id, "assistant", full_answer, sources)
