import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Nạp lại vector DB đã lưu ở bước ingest
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Câu hỏi test — sửa lại theo nội dung file PDF của bạn
question = "Nội dung chính của tài liệu này là gì?"

# Semantic search: lấy 3 đoạn liên quan nhất
results = vectorstore.similarity_search(question, k=3)
print("--- 3 đoạn liên quan nhất tìm được ---")
for i, doc in enumerate(results):
    print(f"[{i+1}] {doc.page_content[:200]}")
    print()

# Ghép context lại thành 1 chuỗi
context = "\n\n".join([doc.page_content for doc in results])

# Gọi Gemini để sinh câu trả lời dựa trên context
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
    """Dựa vào ngữ cảnh sau đây, hãy trả lời câu hỏi bằng tiếng Việt.
Nếu ngữ cảnh không đủ thông tin để trả lời, hãy nói rõ là không tìm thấy thông tin.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""
)

chain = prompt | llm
response = chain.invoke({"context": context, "question": question})

print("--- Câu trả lời ---")
print(response.content[0]['text'])