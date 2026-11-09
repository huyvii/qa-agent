from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import pytest

import rag_core


@pytest.fixture(autouse=True)
def _clean_bm25_cache():
    rag_core.clear_retrieval_cache()
    yield
    rag_core.clear_retrieval_cache()


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
        self.seen_k = []
        self.get_calls = 0

    def similarity_search(self, question, k, filter=None):
        self.seen_filters.append(filter)
        self.seen_k.append(k)
        return self.vec_ranked[:k]

    def get(self, where=None, include=None):
        self.get_calls += 1
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


CORPUS = [_doc(text, page) for page, text in enumerate(MATH_EXERCISES[:3], start=1)]


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


def test_retrieve_bm25_mode_ranks_by_keywords_without_vector_call():
    vs = _FakeVectorStore(vec_ranked=[CORPUS[0]], corpus=CORPUS)

    results = rag_core.retrieve(
        "dãy số cộng công sai", vs, source_file_filter="math_exercises.md", k=1, mode="bm25"
    )

    assert "số hạng thứ 10" in results[0].page_content
    assert vs.seen_k == []  # nhánh vector không bị gọi khi mode="bm25"


def test_retrieve_falls_back_to_vector_when_corpus_empty():
    vs = _FakeVectorStore(vec_ranked=[CORPUS[0], CORPUS[1]], corpus=[])

    results = rag_core.retrieve("equation", vs, source_file_filter="math_exercises.md", k=1)

    assert results[0].page_content == CORPUS[0].page_content


def test_bm25_corpus_is_cached_until_cleared():
    vs = _FakeVectorStore(vec_ranked=CORPUS, corpus=CORPUS)

    rag_core.retrieve("chu vi", vs, source_file_filter="math_exercises.md", k=2)
    rag_core.retrieve("phương trình", vs, source_file_filter="math_exercises.md", k=2)
    assert vs.get_calls == 1

    rag_core.clear_retrieval_cache()
    rag_core.retrieve("dãy số", vs, source_file_filter="math_exercises.md", k=2)
    assert vs.get_calls == 2


class _FakeReranker:
    """Cross-encoder giả: chấm điểm theo nội dung đoạn để kiểm soát thứ hạng."""

    def __init__(self, score_fn):
        self.score_fn = score_fn

    def predict(self, pairs):
        return [self.score_fn(doc_text) for _, doc_text in pairs]


def test_retrieve_reranker_reorders_fused_candidates_and_fetches_full_pool():
    vs = _FakeVectorStore(vec_ranked=[CORPUS[0], CORPUS[1], CORPUS[2]], corpus=CORPUS)
    reranker = _FakeReranker(
        lambda text: 3 if "hình chữ nhật" in text.lower() else (2 if "dãy số" in text.lower() else 1)
    )

    results = rag_core.retrieve(
        "math exercise", vs, source_file_filter="math_exercises.md", k=1, reranker=reranker
    )

    assert [d.page_content for d in results] == [CORPUS[1].page_content]
    # Ngay cả khi k=1, nhánh tìm kiếm vẫn phải lấy tối thiểu RERANK_POOL_SIZE ứng viên
    assert vs.seen_k == [rag_core.RERANK_POOL_SIZE]


def test_bm25_handles_two_chunk_corpus_without_zero_idf():
    # Với N=2, BM25Okapi gốc cho IDF=0 khi từ xuất hiện ở 1/2 đoạn; _LuceneBM25 phải xếp đúng đoạn khớp
    two_docs = [CORPUS[0], CORPUS[1]]
    vs = _FakeVectorStore(vec_ranked=two_docs, corpus=two_docs)

    results = rag_core.retrieve(
        "chiều rộng diện tích", vs, source_file_filter="math_exercises.md", k=1, mode="bm25"
    )
    assert results[0].page_content == CORPUS[1].page_content


def test_condense_question_leaves_standalone_questions_untouched():
    history = [
        {"role": "user", "content": "Nghiệm của phương trình 3x + 5 = 26 là gì?"},
        {"role": "assistant", "content": "x = 7."},
    ]

    # Câu hỏi độc lập ngắn (5-7 từ) đổi chủ đề không được ghép câu cũ vào
    q1 = "Hạn nộp hồ sơ khi nào?"
    assert rag_core.condense_question(q1, history) == q1

    # Từ "còn" nằm giữa câu độc lập không phải câu nối tiếp
    q2 = "Trong báo cáo quý ba còn khoản nợ nào chưa thanh toán không?"
    assert rag_core.condense_question(q2, history) == q2


def test_condense_question_expands_follow_ups_and_preserves_anchor_across_turns():
    anchor = "Hình chữ nhật có chu vi 36 cm và chiều dài 11 cm thì chiều rộng là bao nhiêu?"
    history = [
        {"role": "user", "content": anchor},
        {"role": "assistant", "content": "Chiều rộng là 7 cm."},
        {"role": "user", "content": "Còn diện tích?"},
        {"role": "assistant", "content": "Diện tích là 77 cm²."},
    ]

    # Ở lượt 3 ("Thế còn nửa chu vi?"), mỏ neo phải giữ được câu hỏi gốc ở lượt 1
    condensed = rag_core.condense_question("Thế còn nửa chu vi?", history)
    assert anchor in condensed
    assert "Thế còn nửa chu vi?" in condensed


def test_build_conversation_memory_combines_older_relevant_and_recent_turns():
    history = []
    # Lượt 1 (cũ) chứa thông tin đặc thù về AdamW
    history.append({"role": "user", "content": "Cấu hình chuẩn của AdamW dùng weight_decay bằng mấy?"})
    history.append({"role": "assistant", "content": "weight_decay = 0.01 và 500 bước warmup."})
    # 5 lượt trung gian khác chủ đề để đẩy lượt AdamW ra khỏi cửa sổ HISTORY_TURNS=4
    for idx in range(5):
        history.append({"role": "user", "content": f"Câu hỏi trung gian số {idx} về hình học phẳng"})
        history.append({"role": "assistant", "content": f"Trả lời trung gian số {idx}."})

    memory = rag_core.build_conversation_memory("Nhắc lại hệ số weight_decay của AdamW", history)

    assert "Các lượt cũ liên quan:" in memory
    assert "weight_decay = 0.01" in memory
    assert "Các lượt gần đây:" in memory


def test_answer_question_deduplicates_one_based_sources_and_hides_out_of_scope():
    # PyPDFLoader đánh số trang từ 0 -> hiển thị phải là trang 1 và không lặp trùng trang
    docs = [
        Document(page_content="Đoạn 1 trang đầu", metadata={"source_file": "doc.pdf", "page": 0}),
        Document(page_content="Đoạn 2 cùng trang đầu", metadata={"source_file": "doc.pdf", "page": 0}),
    ]
    vs = _FakeVectorStore(vec_ranked=docs, corpus=docs)

    in_scope_llm = RunnableLambda(lambda _inp: SimpleNamespace(content="Câu trả lời từ PDF."))
    answer, sources = rag_core.answer_question("Câu hỏi bất kỳ", vs, in_scope_llm, source_file_filter="doc.pdf", k=2)
    assert answer == "Câu trả lời từ PDF."
    assert sources == ["doc.pdf - trang 1"]

    out_scope_llm = RunnableLambda(
        lambda _inp: SimpleNamespace(content="Ngoài phạm vi tài liệu: Câu trả lời theo kiến thức chung.")
    )
    _, out_sources = rag_core.answer_question("Thủ đô Pháp là gì?", vs, out_scope_llm, source_file_filter="doc.pdf", k=2)
    assert out_sources == []


def test_create_llm_validates_api_key_and_retries(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        rag_core.create_llm(api_key="")

    with pytest.raises(RuntimeError, match="GEMINI_MAX_RETRIES"):
        rag_core.create_llm(api_key="dummy-key", max_retries=-1)
