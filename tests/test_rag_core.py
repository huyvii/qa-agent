from langchain_core.documents import Document

import rag_core


def test_tokenize_handles_vietnamese_and_numbers():
    tokens = rag_core._tokenize("Chức danh: GPA 7.69!")

    assert "chức" in tokens and "danh" in tokens
    assert "gpa" in tokens and "7" in tokens and "69" in tokens


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


def _doc(text):
    return Document(page_content=text, metadata={"source_file": "a.pdf"})


CORPUS = [
    _doc("An Pham Hoang Robotics AI Engineer GPA 7.69"),
    _doc("master thesis From Video to Simulation Real2Sim"),
    _doc("bachelor thesis Tele-Manipulator remote control robot"),
]


def test_retrieve_hybrid_prefers_docs_hit_by_both_branches():
    # Vector xếp A trước, nhưng BM25 khớp C ("bachelor thesis") mạnh nhất
    # -> C (được cả hai nhánh chọn) phải đứng đầu kết quả hợp nhất
    vs = _FakeVectorStore(vec_ranked=[CORPUS[0], CORPUS[2], CORPUS[1]], corpus=CORPUS)

    results = rag_core.retrieve(
        "bachelor thesis về chủ đề gì", vs, source_file_filter="a.pdf", k=3
    )

    assert results[0].page_content == CORPUS[2].page_content
    assert vs.seen_filters == [{"source_file": "a.pdf"}]


def test_retrieve_non_hybrid_returns_vector_order():
    vs = _FakeVectorStore(vec_ranked=[CORPUS[1], CORPUS[0], CORPUS[2]], corpus=CORPUS)

    results = rag_core.retrieve(
        "gì đó", vs, source_file_filter="a.pdf", k=2, hybrid=False
    )

    assert [d.page_content for d in results] == [
        CORPUS[1].page_content,
        CORPUS[0].page_content,
    ]


def test_retrieve_falls_back_to_vector_when_corpus_empty():
    vs = _FakeVectorStore(vec_ranked=[CORPUS[0], CORPUS[1]], corpus=[])

    results = rag_core.retrieve("bachelor", vs, source_file_filter="a.pdf", k=1)

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
        lambda text: 3 if "Tele-Manipulator" in text else (2 if "thesis" in text else 1)
    )

    results = rag_core.retrieve(
        "bachelor thesis", vs, source_file_filter="a.pdf", k=2, reranker=reranker
    )

    assert [d.page_content for d in results] == [
        CORPUS[2].page_content,
        CORPUS[1].page_content,
    ]
