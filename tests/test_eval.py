import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from retrieval_baseline import (
    TEST_CASES,
    compute_rank_metrics,
    load_fixture_documents,
    normalize,
)


def test_all_project_entrypoints_compile_without_syntax_errors():
    for rel_path in [
        "app.py",
        "chat_db.py",
        "rag_core.py",
        "ingest.py",
        "query.py",
        "evaluation/retrieval_baseline.py",
    ]:
        py_compile.compile(str(ROOT / rel_path), doraise=True)


def test_normalize_is_case_insensitive():
    assert normalize("Area: 77 cm²") == normalize("area: 77 cm²")


def test_normalize_collapses_whitespace():
    assert normalize("a  b\n\tc") == "a b c"


def test_normalize_nfkc_unifies_unicode_forms():
    # Dấu cách không ngắt (NBSP) và ký tự tổ hợp phải quy về cùng một dạng
    assert normalize("Area\u00a077 cm²") == normalize("Area 77 cm²")
    assert normalize("e\u0301") == normalize("é")


def test_fixture_has_non_trivial_corpus_and_valid_reference_evidence():
    docs = load_fixture_documents()
    # Corpus phải đủ lớn (>> k=3) để phép đo Hit@3 có ý nghĩa phân loại thực sự
    assert len(docs) >= 20
    by_page = {doc.metadata["page"]: doc.page_content for doc in docs}
    for case in TEST_CASES:
        ref_text = by_page[case["reference_page"]]
        assert normalize(case["expected_evidence"]) in normalize(ref_text), case["id"]


def test_compute_rank_metrics_hit_and_mrr():
    # 4 truy vấn: trúng hạng 1, trúng hạng 3, trượt (None), trúng hạng 2
    metrics = compute_rank_metrics([1, 3, None, 2], k=3)
    assert metrics["hit_at_1"] == 1
    assert metrics["hits"] == 3
    assert metrics["accuracy"] == 0.75
    expected_mrr = round((1.0 + 1.0 / 3.0 + 0.0 + 0.5) / 4.0, 4)
    assert metrics["mrr"] == expected_mrr
