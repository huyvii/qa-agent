import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# 1. Đọc file PDF
loader = PyPDFLoader("data/Hoang-An_Pham-AI-Engineer (2).pdf")
documents = loader.load()
print(f"Đã đọc {len(documents)} trang từ PDF")

# 2. Chia nhỏ văn bản thành chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Đã chia thành {len(chunks)} chunk")
print("--- Ví dụ chunk đầu tiên ---")
print(chunks[0].page_content[:300])

# 3. Tạo embedding model (chạy local, free, CPU)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Lưu vào Chroma vector DB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
print(f"Đã lưu {len(chunks)} vector vào Chroma DB tại ./chroma_db")