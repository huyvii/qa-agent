from langchain_core.documents import Document
from pathlib import Path

import rag_core


MATH_EXERCISES = [
    section.strip()
    for section in (Path(__file__).parent / "fixtures" / "math_exercises.md")
    .read_text(encoding="utf-8")
    .split("\n\n")
    if section.strip() and not section.startswith("#")
]


def test_tokenize_handles_vietnamese_and_numbers():
    tokens = rag_core._tokenize("Diện tích: 77 cm2!")

    assert "diện" in tokens and "tích" in tokens
    assert "77" in tokens and "cm2" in tokens


def test_extract_text_passthrough_string():
    assert rag_core._extract_text("abc") == "abc"


def test_extract_text_joins_text_blocks_only():
    content = [
        {"type": "text", "text": "Phần A"},
        {"type": "image"},
        {"type": "text", "text": "Phần B"},
    ]

    assert rag_core._extract_text(content) == "Phần APhần B"


class _FakeVectorStore:
    """Mô phỏng đúng interface mà rag_core.retrieve cần: similarity_search + get."""

    def __init__(self, vec_ranked, corpus):
        self.vec_ranked = vec_ranked
        self.corpus = corpus
        self.seen_filters = []

    def similarity_search(self, question, k, filter=None):
        self.seen_filters.append(filter)
        return self.vec_ranked[:k]

    def get(self, where=None, include=None):
        return {
            "ids": [f"id{i}" for i in range(len(self.corpus))],
            "documents": [d.page_content for d in self.corpus],
            "metadatas": [d.metadata for d in self.corpus],
        }


def _doc(text, page):
    return Document(
        page_content=text,
        metadata={"source_file": "math_exercises.md", "page": page},
    )


CORPUS = [_doc(text, page + 1) for page, text in enumerate(MATH_EXERCISES)]


def test_retrieve_hybrid_fuses_vector_and_keyword_results():
    # Vector xếp dãy số học trước; BM25 khớp bài hình chữ nhật trước.
    # Tài liệu hình chữ nhật được cả hai nhánh chọn nên đứng đầu sau RRF.
    vs = _FakeVectorStore(vec_ranked=[CORPUS[2], CORPUS[1], CORPUS[0]], corpus=CORPUS)

    results = rag_core.retrieve(
        "hình chữ nhật chu vi", vs, source_file_filter="math_exercises.md", k=3
    )

    assert "chiều rộng" in results[0].page_content
    assert vs.seen_filters == [{"source_file": "math_exercises.md"}]


def test_retrieve_non_hybrid_returns_vector_order():
    vs = _FakeVectorStore(vec_ranked=[CORPUS[1], CORPUS[0], CORPUS[2]], corpus=CORPUS)

    results = rag_core.retrieve(
        "phương trình", vs, source_file_filter="math_exercises.md", k=2, hybrid=False
    )

    assert [d.page_content for d in results] == [
        CORPUS[1].page_content,
        CORPUS[0].page_content,
    ]


def test_retrieve_falls_back_to_vector_when_corpus_empty():
    vs = _FakeVectorStore(vec_ranked=[CORPUS[0], CORPUS[1]], corpus=[])

    results = rag_core.retrieve("equation", vs, source_file_filter="math_exercises.md", k=1)

    assert results[0].page_content == CORPUS[0].page_content


class _FakeReranker:
    """Cross-encoder giả: chấm điểm theo nội dung đoạn để kiểm soát thứ hạng."""

    def __init__(self, score_fn):
        self.score_fn = score_fn

    def predict(self, pairs):
        return [self.score_fn(doc_text) for _, doc_text in pairs]


def test_retrieve_reranker_reorders_fused_candidates():
    # Vector xếp C cuối cùng, nhưng reranker chấm C cao nhất -> C phải lên đầu
    vs = _FakeVectorStore(vec_ranked=[CORPUS[0], CORPUS[1], CORPUS[2]], corpus=CORPUS)
    reranker = _FakeReranker(
        lambda text: 3 if "hình chữ nhật" in text else (2 if "dãy số" in text else 1)
    )

    results = rag_core.retrieve(
        "math exercise", vs, source_file_filter="math_exercises.md", k=2, reranker=reranker
    )

    assert [d.page_content for d in results] == [
        CORPUS[1].page_content,
        CORPUS[2].page_content,
    ]
