"""Measure Hit@k using a self-contained math exercise fixture.

By default, the fixture is indexed into a temporary Chroma collection, so the
evaluation does not depend on a user's ignored local database or call Gemini.
Pass --chroma-dir and --source-file to evaluate an existing collection instead.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

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
        "question": "Nghiệm của phương trình 3x + 5 = 26 là gì?",
        "expected_evidence": "x = 7",
        "reference_page": 1,
    },
    {
        "id": "rectangle_width",
        "question": "Chiều rộng hình chữ nhật chu vi 36 cm, chiều dài 11 cm là bao nhiêu?",
        "expected_evidence": "chiều rộng là 7 cm",
        "reference_page": 2,
    },
    {
        "id": "rectangle_area",
        "question": "Diện tích hình chữ nhật là bao nhiêu?",
        "expected_evidence": "diện tích là 77 cm²",
        "reference_page": 2,
    },
    {
        "id": "arithmetic_sequence_term",
        "question": "Số hạng thứ 10 của dãy số cộng là bao nhiêu?",
        "expected_evidence": "số hạng thứ 10 là 2 + 9 × 3 = 29",
        "reference_page": 3,
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


def _score_cases(vectorstore, source_file: str, k: int, reranker=None) -> dict:
    cases = []
    for case in TEST_CASES:
        documents = retrieve(
            case["question"], vectorstore, source_file, k=k, reranker=reranker
        )
        expected = normalize(case["expected_evidence"])
        retrieved = [
            {
                "page": document.metadata.get("page"),
                "preview": document.page_content[:180],
                "contains_expected_evidence": expected in normalize(document.page_content),
            }
            for document in documents
        ]
        cases.append(
            {
                **case,
                "hit": any(item["contains_expected_evidence"] for item in retrieved),
                "retrieved_chunks": retrieved,
            }
        )

    hits = sum(case["hit"] for case in cases)
    return {
        "metric": f"retrieval_hit_at_{k}",
        "retrieval": "hybrid: multilingual vector + BM25, fused by RRF (rag_core.retrieve)",
        "reranker": rag_core.RERANKER_MODEL_NAME if reranker is not None else None,
        "definition": (
            f"A case is a hit when one of the top {k} retrieved chunks contains its "
            "predefined expected evidence."
        ),
        "fixture": str(FIXTURE_PATH.relative_to(ROOT)),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "source_file": source_file,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(cases),
        "hits": hits,
        "accuracy": hits / len(cases),
        "cases": cases,
    }


def evaluate(
    source_file: str | None = None,
    chroma_dir: Path | None = None,
    k: int = 3,
    reranker=None,
) -> dict:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if chroma_dir is None:
        source_file = source_file or DEFAULT_SOURCE_FILE
        vectorstore = Chroma.from_documents(
            documents=load_fixture_documents(source_file),
            embedding=embeddings,
        )
        try:
            return _score_cases(vectorstore, source_file, k, reranker)
        finally:
            rag_core.clear_retrieval_cache()

    source_file = source_file or DEFAULT_SOURCE_FILE
    vectorstore = Chroma(
        persist_directory=str(chroma_dir), embedding_function=embeddings
    )
    return _score_cases(vectorstore, source_file, k, reranker)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure math exercise retrieval Hit@k.")
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
        "--rerank",
        action="store_true",
        help="Apply the cross-encoder reranker from rag_core.get_reranker().",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.chroma_dir is not None and not args.chroma_dir.exists():
        parser.error(f"Chroma directory does not exist: {args.chroma_dir}")

    reranker = rag_core.get_reranker() if args.rerank else None
    result = evaluate(args.source_file, args.chroma_dir, args.k, reranker=reranker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Retrieval Hit@{args.k}: {result['hits']}/{result['total_cases']} "
        f"({result['accuracy']:.0%})"
    )
    print(f"Saved evaluation to {args.output}")


if __name__ == "__main__":
    main()
