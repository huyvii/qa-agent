import sqlite3
import json
from datetime import datetime
from typing import Optional

DB_PATH = "chat_history.db"


def _now() -> str:
    """Timestamp ISO 8601 với microsecond LUÔN đủ 6 chữ số.
    Không dùng thẳng datetime.now().isoformat() vì nó bỏ phần .000000
    khi microsecond ngẫu nhiên = 0, làm chuỗi ngắn hơn bình thường và
    có thể sort sai khi ORDER BY created_at (so sánh dạng text)."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Tạo các bảng nếu chưa tồn tại. Gọi 1 lần lúc app khởi động."""
    conn = get_connection()
    cur = conn.cursor()

    # Bảng theo dõi các file đã được ingest vào Chroma (tránh nhúng trùng lặp)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            source_file TEXT PRIMARY KEY,
            chunk_count INTEGER NOT NULL,
            ingested_at TEXT NOT NULL
        )
    """)

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
    conn.close()


# ==================== documents (tracking file đã ingest) ====================

def document_exists(source_file: str) -> bool:
    """Kiểm tra file này đã từng ingest vào Chroma chưa (theo tên file)."""
    conn = get_connection()
    row = conn.execute(
        "SELECT 1 FROM documents WHERE source_file = ?", (source_file,)
    ).fetchone()
    conn.close()
    return row is not None


def add_document(source_file: str, chunk_count: int):
    """Ghi nhận 1 file đã ingest xong. Gọi ngay sau khi ingest_pdf() chạy thành công."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO documents (source_file, chunk_count, ingested_at) VALUES (?, ?, ?)",
        (source_file, chunk_count, _now())
    )
    conn.commit()
    conn.close()


def get_documents() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ==================== conversations ====================

def create_conversation(source_file: str, title: Optional[str] = None) -> int:
    """Tạo 1 conversation mới, gắn với đúng 1 source_file. Trả về id vừa tạo."""
    if title is None:
        title = source_file
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO conversations (title, source_file, created_at) VALUES (?, ?, ?)",
        (title, source_file, _now())
    )
    conn.commit()
    conv_id = cur.lastrowid
    conn.close()
    return conv_id


def get_conversations() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM conversations ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation(conversation_id: int) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_conversation(conversation_id: int):
    """Xóa conversation + toàn bộ message của nó.
    KHÔNG đụng tới bảng documents hay Chroma DB — dữ liệu PDF vẫn dùng được
    cho các conversation khác đang gắn với cùng file đó."""
    conn = get_connection()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()


# ==================== messages ====================

def save_message(conversation_id: int, role: str, content: str, sources: Optional[list] = None):
    conn = get_connection()
    sources_json = json.dumps(sources) if sources else None
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (conversation_id, role, content, sources_json, _now())
    )
    conn.commit()
    conn.close()


def get_messages(conversation_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,)
    ).fetchall()
    conn.close()

    messages = []
    for r in rows:
        msg = dict(r)
        msg["sources"] = json.loads(msg["sources"]) if msg["sources"] else []
        messages.append(msg)
    return messages