"""Measure retrieval Hit@1, Hit@k, MRR@k, and latency on a 24-chunk bilingual STEM/ML fixture.

By default, the bundled fixture (Vietnamese + English technical passages with
distractors) is indexed into an ephemeral Chroma collection, so evaluation runs
reproducibly without a user's private PDF or Gemini API calls.
Use --compare to benchmark BM25, Multilingual Vector, Hybrid RRF, and
Hybrid + Cross-Encoder Reranker side by side.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rag_core  # noqa: E402
from rag_core import retrieve  # noqa: E402

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "math_exercises.md"
DEFAULT_SOURCE_FILE = "math_exercises.md"
DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "retrieval_baseline.json"

TEST_CASES = [
    {
        "id": "linear_equation_solution",
        "category": "exact_math",
        "question": "Nghiệm của phương trình 3x + 5 = 26 là gì?",
        "expected_evidence": "x = 7",
        "reference_page": 1,
    },
    {
        "id": "rectangle_area_disambiguation",
        "category": "distractor_math",
        "question": "Hình chữ nhật có chu vi 36 cm và chiều dài 11 cm thì diện tích bằng bao nhiêu?",
        "expected_evidence": "diện tích là 77 cm²",
        "reference_page": 2,
    },
    {
        "id": "bayes_posterior_probability",
        "category": "paraphrase_stats",
        "question": "Nếu xét nghiệm dương tính với bệnh có tỷ lệ nền 1% thì khả năng mắc bệnh thực tế là bao nhiêu?",
        "expected_evidence": "16.6%",
        "reference_page": 7,
    },
    {
        "id": "pca_components_95_variance",
        "category": "exact_param",
        "question": "Cần giữ bao nhiêu thành phần chính trong PCA để đạt ít nhất 95% phương sai?",
        "expected_evidence": "k = 42",
        "reference_page": 8,
    },
    {
        "id": "adamw_hyperparameters",
        "category": "distractor_optim",
        "question": "Hệ số weight decay và số bước warmup trong cấu hình chuẩn của AdamW là bao nhiêu?",
        "expected_evidence": "weight_decay = 0.01",
        "reference_page": 11,
    },
    {
        "id": "label_smoothing_probability",
        "category": "exact_param",
        "question": "Khi dùng Label Smoothing ε = 0.1 cho 10 lớp thì nhãn đúng nhận giá trị xác suất nào?",
        "expected_evidence": "0.91",
        "reference_page": 13,
    },
    {
        "id": "attention_scaling_factor",
        "category": "cross_lingual_vi_en",
        "question": "Tại sao tích vô hướng trong cơ chế attention lại phải chia cho căn bậc hai số chiều khóa?",
        "expected_evidence": "vanishingly small gradients",
        "reference_page": 15,
    },
    {
        "id": "lora_parameter_reduction",
        "category": "cross_lingual_vi_en",
        "question": "Phương pháp thích nghi hạng thấp đóng băng trọng số gốc giảm được bao nhiêu phần trăm tham số huấn luyện?",
        "expected_evidence": "over 99%",
        "reference_page": 16,
    },
    {
        "id": "cross_encoder_joint_attention",
        "category": "cross_lingual_vi_en",
        "question": "Mô hình nào ghép chung câu hỏi và văn bản để tính tự chú ý xuyên suốt các tầng thay vì mã hóa độc lập?",
        "expected_evidence": "Cross-encoders concatenate query and passage",
        "reference_page": 17,
    },
    {
        "id": "sram_tiling_io_awareness",
        "category": "cross_lingual_vi_en",
        "question": "Kỹ thuật chia khối đưa vào bộ nhớ đệm trên chip giúp tránh ghi ma trận trung gian kích thước bậc hai xuống bộ nhớ băng thông cao như thế nào?",
        "expected_evidence": "on-chip SRAM",
        "reference_page": 18,
    },
    {
        "id": "rrf_smoothing_constant",
        "category": "paraphrase_ir",
        "question": "Hằng số làm mượt trong công thức hợp nhất thứ hạng nghịch đảo thường đặt bằng mấy?",
        "expected_evidence": "k_rrf = 60",
        "reference_page": 20,
    },
    {
        "id": "imbalanced_classification_metric",
        "category": "conceptual_ml",
        "question": "Khi lớp thiểu số chỉ chiếm 0.5% dữ liệu thì nên dùng thước đo nào thay cho Accuracy?",
        "expected_evidence": "PR-AUC",
        "reference_page": 23,
    },
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def load_fixture_documents(source_file: str = DEFAULT_SOURCE_FILE) -> list[Document]:
    sections = [
        section.strip()
        for section in FIXTURE_PATH.read_text(encoding="utf-8").split("\n\n")
        if section.strip() and not section.startswith("#")
    ]
    return [
        Document(
            page_content=section,
            metadata={"source_file": source_file, "page": page},
        )
        for page, section in enumerate(sections, start=1)
    ]


def compute_rank_metrics(hit_ranks: list[int | None], k: int) -> dict:
    """Compute Hit@1, Hit@k, and MRR@k from 1-based hit ranks (None if missed in top-k)."""
    total = len(hit_ranks)
    if total == 0:
        return {"hit_at_1": 0, "hits": 0, "accuracy": 0.0, "hit_rate_at_1": 0.0, "mrr": 0.0}
    hits_at_1 = sum(1 for r in hit_ranks if r == 1)
    hits_at_k = sum(1 for r in hit_ranks if r is not None and r <= k)
    mrr = sum((1.0 / r) for r in hit_ranks if r is not None and r <= k) / total
    return {
        "hit_at_1": hits_at_1,
        "hit_rate_at_1": hits_at_1 / total,
        "hits": hits_at_k,
        "accuracy": hits_at_k / total,
        "mrr": round(mrr, 4),
    }


def _score_cases(
    vectorstore,
    source_file: str,
    k: int,
    mode: str = "hybrid",
    reranker=None,
    corpus_size: int | None = None,
) -> dict:
    cases = []
    hit_ranks: list[int | None] = []
    latencies_ms: list[float] = []

    # Warm up BM25 cache once so per-query latency reflects steady-state serving
    if mode in {"bm25", "hybrid"}:
        retrieve(TEST_CASES[0]["question"], vectorstore, source_file, k=k, mode=mode)

    for case in TEST_CASES:
        started = time.perf_counter()
        documents = retrieve(
            case["question"],
            vectorstore,
            source_file,
            k=k,
            mode=mode,
            reranker=reranker,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)

        expected = normalize(case["expected_evidence"])
        retrieved = []
        first_hit_rank: int | None = None
        for rank, document in enumerate(documents, start=1):
            matched = expected in normalize(document.page_content)
            if matched and first_hit_rank is None:
                first_hit_rank = rank
            retrieved.append(
                {
                    "rank": rank,
                    "page": document.metadata.get("page"),
                    "preview": document.page_content[:180],
                    "contains_expected_evidence": matched,
                }
            )

        hit_ranks.append(first_hit_rank)
        cases.append(
            {
                **case,
                "hit": first_hit_rank is not None,
                "hit_at_1": first_hit_rank == 1,
                "hit_rank": first_hit_rank,
                "latency_ms": round(elapsed_ms, 2),
                "retrieved_chunks": retrieved,
            }
        )

    metrics = compute_rank_metrics(hit_ranks, k)
    avg_latency_ms = round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0.0
    label = f"{mode}+rerank" if reranker is not None else mode
    reranker_name = getattr(reranker, "model_name", None) or (
        rag_core.RERANKER_MODEL_NAME if reranker is not None else None
    )

    return {
        "metric": f"retrieval_hit_at_{k}",
        "mode": label,
        "retrieval_branch": mode,
        "reranker": reranker_name,
        "definition": (
            f"A case is a hit when one of the top {k} retrieved chunks contains its "
            "predefined expected evidence."
        ),
        "fixture": str(FIXTURE_PATH.relative_to(ROOT)),
        "corpus_chunks": corpus_size,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "source_file": source_file,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(cases),
        "hit_at_1": metrics["hit_at_1"],
        "hit_rate_at_1": metrics["hit_rate_at_1"],
        "hits": metrics["hits"],
        "accuracy": metrics["accuracy"],
        "mrr": metrics["mrr"],
        "avg_latency_ms": avg_latency_ms,
        "cases": cases,
    }


def _open_or_build_vectorstore(
    source_file: str | None,
    chroma_dir: Path | None,
) -> tuple[Chroma, str, int | None, bool]:
    embeddings = rag_core.get_embedding_model()
    resolved_source = source_file or DEFAULT_SOURCE_FILE
    if chroma_dir is None:
        docs = load_fixture_documents(resolved_source)
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
        return vectorstore, resolved_source, len(docs), True
    vectorstore = Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
    )
    stored = vectorstore.get(where={"source_file": resolved_source}, include=[])
    count = len(stored.get("ids") or [])
    return vectorstore, resolved_source, count, False


def evaluate(
    source_file: str | None = None,
    chroma_dir: Path | None = None,
    k: int = 3,
    reranker=None,
    mode: str = "hybrid",
) -> dict:
    vectorstore, resolved_source, corpus_size, is_ephemeral = _open_or_build_vectorstore(
        source_file, chroma_dir
    )
    try:
        return _score_cases(
            vectorstore,
            resolved_source,
            k=k,
            mode=mode,
            reranker=reranker,
            corpus_size=corpus_size,
        )
    finally:
        if is_ephemeral:
            rag_core.clear_retrieval_cache()


def evaluate_comparison(
    source_file: str | None = None,
    chroma_dir: Path | None = None,
    k: int = 3,
    reranker_model: str | None = None,
) -> dict:
    """Run all 4 retrieval stages on the exact same indexed collection."""
    vectorstore, resolved_source, corpus_size, is_ephemeral = _open_or_build_vectorstore(
        source_file, chroma_dir
    )
    active_model_name = reranker_model or rag_core.RERANKER_MODEL_NAME
    reranker = (
        rag_core.CrossEncoder(active_model_name)
        if reranker_model
        else rag_core.get_reranker()
    )
    reranker.model_name = active_model_name
    configs = [
        ("bm25", "BM25 (lexical)", None),
        ("vector", "Multilingual Vector", None),
        ("hybrid", "Hybrid (Vector + BM25 + RRF)", None),
        ("hybrid", "Hybrid + Cross-Encoder Reranker", reranker),
    ]
    try:
        runs = []
        for branch_mode, display_name, active_reranker in configs:
            report = _score_cases(
                vectorstore,
                resolved_source,
                k=k,
                mode=branch_mode,
                reranker=active_reranker,
                corpus_size=corpus_size,
            )
            report["display_name"] = display_name
            runs.append(report)
        return {
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "fixture": str(FIXTURE_PATH.relative_to(ROOT)),
            "corpus_chunks": corpus_size,
            "total_cases": len(TEST_CASES),
            "k": k,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_model": active_model_name,
            "runs": runs,
        }
    finally:
        if is_ephemeral:
            rag_core.clear_retrieval_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure retrieval Hit@1, Hit@k, MRR@k, and latency.")
    parser.add_argument(
        "--source-file",
        help="source_file metadata to filter when using an existing Chroma collection",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        help="evaluate an existing Chroma collection instead of the bundled fixture",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=["bm25", "vector", "hybrid"],
        default="hybrid",
        help="retrieval branch mode before optional reranking",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="apply the cross-encoder reranker from rag_core.get_reranker()",
    )
    parser.add_argument(
        "--reranker-model",
        help="override CrossEncoder model name for A/B benchmarking",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="benchmark BM25, Vector, Hybrid RRF, and Hybrid + Reranker side by side",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.chroma_dir is not None and not args.chroma_dir.exists():
        parser.error(f"Chroma directory does not exist: {args.chroma_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.compare:
        comparison = evaluate_comparison(
            args.source_file,
            args.chroma_dir,
            args.k,
            reranker_model=args.reranker_model,
        )
        args.output.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"Corpus: {comparison['corpus_chunks']} chunks | "
            f"Queries: {comparison['total_cases']} | Top-k: {args.k}\n"
        )
        header = f"{'Configuration':<34} {'Hit@1':>9} {f'Hit@{args.k}':>9} {f'MRR@{args.k}':>8} {'Avg ms':>9}"
        print(header)
        print("-" * len(header))
        for run in comparison["runs"]:
            n = run["total_cases"]
            h1 = f"{run['hit_at_1']}/{n}"
            hk = f"{run['hits']}/{n}"
            print(
                f"{run['display_name']:<34} {h1:>9} {hk:>9} "
                f"{run['mrr']:>8.3f} {run['avg_latency_ms']:>8.1f}"
            )
        print(f"\nSaved comparison report to {args.output}")
        return

    reranker = None
    if args.rerank or args.reranker_model:
        reranker = (
            rag_core.CrossEncoder(args.reranker_model)
            if args.reranker_model
            else rag_core.get_reranker()
        )
        if args.reranker_model:
            reranker.model_name = args.reranker_model
    result = evaluate(
        args.source_file,
        args.chroma_dir,
        args.k,
        reranker=reranker,
        mode=args.mode,
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    n = result["total_cases"]
    print(
        f"[{result['mode']}] Corpus: {result['corpus_chunks']} chunks | "
        f"Hit@1: {result['hit_at_1']}/{n} ({result['hit_rate_at_1']:.0%}) | "
        f"Hit@{args.k}: {result['hits']}/{n} ({result['accuracy']:.0%}) | "
        f"MRR@{args.k}: {result['mrr']:.3f} | "
        f"Avg: {result['avg_latency_ms']:.1f} ms/query"
    )
    print(f"Saved evaluation to {args.output}")


if __name__ == "__main__":
    main()
