from contextlib import closing
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
from typing import Optional

# Đường dẫn có thể override qua env — cần cho Docker volume và unit test
PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv("CHAT_DB_PATH", str(PROJECT_DIR / "chat_history.db"))


def _now() -> str:
    """Timestamp ISO 8601 với microsecond LUÔN đủ 6 chữ số.
    Không dùng thẳng datetime.now().isoformat() vì nó bỏ phần .000000
    khi microsecond ngẫu nhiên = 0, làm chuỗi ngắn hơn bình thường và
    có thể sort sai khi ORDER BY created_at (so sánh dạng text)."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Tạo các bảng nếu chưa tồn tại. Gọi 1 lần lúc app khởi động."""
    with closing(get_connection()) as conn:
        cur = conn.cursor()

        # Bảng theo dõi các file đã được ingest vào Chroma (tránh nhúng trùng lặp).
        # content_hash để phân biệt "trùng tên nhưng khác nội dung" — trường hợp phải re-ingest.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                source_file TEXT PRIMARY KEY,
                chunk_count INTEGER NOT NULL,
                ingested_at TEXT NOT NULL,
                content_hash TEXT
            )
        """)
        # Migration cho DB tạo trước khi có cột content_hash
        cols = [r["name"] for r in cur.execute("PRAGMA table_info(documents)").fetchall()]
        if "content_hash" not in cols:
            cur.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")

        # Mỗi conversation gắn với đúng 1 source_file -> dùng để filter khi search
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                source_file TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()


# ==================== documents (tracking file đã ingest) ====================

def get_document(source_file: str) -> Optional[dict]:
    """Lấy thông tin 1 file đã ingest (kèm content_hash)."""
    with closing(get_connection()) as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE source_file = ?", (source_file,)
        ).fetchone()
    return dict(row) if row else None


def save_document(source_file: str, content_hash: str, chunk_count: int):
    """Ghi nhận 1 file đã ingest xong. Gọi ngay sau khi ingest_pdf() chạy thành công."""
    with closing(get_connection()) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (source_file, chunk_count, ingested_at, content_hash) VALUES (?, ?, ?, ?)",
            (source_file, chunk_count, _now(), content_hash),
        )
        conn.commit()


def get_documents() -> list[dict]:
    with closing(get_connection()) as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()
    return [dict(r) for r in rows]


def delete_document(source_file: str):
    """Delete a document record and every conversation/message attached to it."""
    with closing(get_connection()) as conn:
        conn.execute(
            "DELETE FROM messages WHERE conversation_id IN "
            "(SELECT id FROM conversations WHERE source_file = ?)",
            (source_file,),
        )
        conn.execute("DELETE FROM conversations WHERE source_file = ?", (source_file,))
        conn.execute("DELETE FROM documents WHERE source_file = ?", (source_file,))
        conn.commit()


# ==================== conversations ====================

def create_conversation(source_file: str, title: Optional[str] = None) -> int:
    """Tạo 1 conversation mới, gắn với đúng 1 source_file. Trả về id vừa tạo."""
    if title is None:
        title = source_file
    with closing(get_connection()) as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title, source_file, created_at) VALUES (?, ?, ?)",
            (title, source_file, _now()),
        )
        conn.commit()
        return cur.lastrowid


def update_conversation_title(conversation_id: int, title: str):
    """Cập nhật tiêu đề hiển thị của cuộc trò chuyện (vd: lấy theo câu hỏi đầu tiên)."""
    cleaned = (title or "").strip()
    if not cleaned:
        return
    with closing(get_connection()) as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (cleaned, conversation_id),
        )
        conn.commit()


def get_conversations() -> list[dict]:
    with closing(get_connection()) as conn:
        rows = conn.execute("SELECT * FROM conversations ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_conversation(conversation_id: int) -> Optional[dict]:
    with closing(get_connection()) as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_conversation(conversation_id: int):
    """Xóa conversation + toàn bộ message của nó.
    KHÔNG đụng tới bảng documents hay Chroma DB — dữ liệu PDF vẫn dùng được
    cho các conversation khác đang gắn với cùng file đó."""
    with closing(get_connection()) as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()


# ==================== messages ====================

def save_message(conversation_id: int, role: str, content: str, sources: Optional[list] = None):
    with closing(get_connection()) as conn:
        sources_json = json.dumps(sources) if sources else None
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, sources_json, _now()),
        )
        conn.commit()


def get_messages(conversation_id: int) -> list[dict]:
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()

    messages = []
    for r in rows:
        msg = dict(r)
        msg["sources"] = json.loads(msg["sources"]) if msg["sources"] else []
        messages.append(msg)
    return messages
