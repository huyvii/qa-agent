import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from retrieval_baseline import normalize


def test_normalize_is_case_insensitive():
    assert normalize("IELTS: 6.5") == normalize("ielts: 6.5")


def test_normalize_collapses_whitespace():
    assert normalize("a  b\n\tc") == "a b c"


def test_normalize_nfkc_unifies_unicode_forms():
    # Dấu cách không ngắt (NBSP) và ký tự tổ hợp phải quy về cùng một dạng
    assert normalize("GPA\u00a07.69") == normalize("GPA 7.69")
    assert normalize("e\u0301") == normalize("é")
