import streamlit as st
import tempfile
import os
from rag_core import ingest_pdf, answer_question, get_vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI
import chat_db

st.set_page_config(page_title="QA Agent", page_icon="📄", layout="wide")

chat_db.init_db()

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

            if chat_db.document_exists(file_name):
                st.info(f"'{file_name}' đã được nạp trước đó — dùng lại dữ liệu cũ, không nhúng lại.")
            else:
                with st.spinner("Đang xử lý tài liệu..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        num_chunks = ingest_pdf(tmp_path, source_file=file_name)
                    finally:
                        os.unlink(tmp_path)

                    chat_db.add_document(file_name, num_chunks)
                    # Reopen the persistent collection after adding new chunks.
                    load_vectorstore.clear()

                st.success(f"Đã nạp xong! ({num_chunks} đoạn văn bản)")

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
        chat_db.save_message(active_id, "user", question)
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm câu trả lời..."):
                vectorstore = load_vectorstore()
                llm = load_llm()
                answer, sources = answer_question(
                    question, vectorstore, llm,
                    source_file_filter=active_conv["source_file"]
                )
            st.write(answer)
            with st.expander("Nguồn tham khảo"):
                for s in sources:
                    st.write(f"- {s}")

        chat_db.save_message(active_id, "assistant", answer, sources)
