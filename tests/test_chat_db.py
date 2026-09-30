import re

import pytest

import chat_db


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_db, "DB_PATH", str(tmp_path / "test.db"))
    chat_db.init_db()
    return chat_db


def test_now_timestamp_always_has_6_digit_microseconds():
    # Sort theo created_at dạng text chỉ đúng khi phần micro luôn đủ 6 chữ số
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}", chat_db._now())


def test_document_roundtrip(tmp_db):
    assert tmp_db.get_document("a.pdf") is None

    tmp_db.save_document("a.pdf", "hash-1", 12)

    doc = tmp_db.get_document("a.pdf")
    assert doc["content_hash"] == "hash-1"
    assert doc["chunk_count"] == 12


def test_save_document_replaces_old_hash(tmp_db):
    tmp_db.save_document("a.pdf", "hash-1", 12)
    tmp_db.save_document("a.pdf", "hash-2", 15)

    doc = tmp_db.get_document("a.pdf")
    assert doc["content_hash"] == "hash-2"
    assert doc["chunk_count"] == 15


def test_conversation_delete_cascades_messages(tmp_db):
    conv_id = tmp_db.create_conversation(source_file="a.pdf")
    tmp_db.save_message(conv_id, "user", "câu hỏi", sources=["a.pdf - trang 0"])
    assert len(tmp_db.get_messages(conv_id)) == 1

    tmp_db.delete_conversation(conv_id)

    assert tmp_db.get_conversation(conv_id) is None
    assert tmp_db.get_messages(conv_id) == []


def test_message_sources_json_roundtrip(tmp_db):
    conv_id = tmp_db.create_conversation(source_file="a.pdf")
    sources = ["a.pdf - trang 1", "a.pdf - trang 2"]

    tmp_db.save_message(conv_id, "assistant", "trả lời", sources=sources)

    msg = tmp_db.get_messages(conv_id)[0]
    assert msg["sources"] == sources
    assert msg["role"] == "assistant"


def test_message_without_sources_gets_empty_list(tmp_db):
    conv_id = tmp_db.create_conversation(source_file="a.pdf")
    tmp_db.save_message(conv_id, "user", "xin chào")

    assert tmp_db.get_messages(conv_id)[0]["sources"] == []
