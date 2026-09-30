import sys
from pathlib import Path

# Cho phép import chat_db / rag_core / evaluation từ gốc repo khi chạy pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
