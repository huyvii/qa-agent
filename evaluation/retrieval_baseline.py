"""Measure retrieval Hit@k for the bundled CV sample.

This evaluates retrieval only: Gemini is deliberately not called. A case is a hit
when any of the top-k chunks contains the predefined evidence text for that case.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SOURCE_FILE = "Hoang-An_Pham-AI-Engineer (2).pdf"
DEFAULT_CHROMA_DIR = Path("chroma_db")
DEFAULT_OUTPUT = Path("evaluation/results/retrieval_baseline.json")

# Each evidence string occurs in the reference chunk listed below. The explicit
# evidence makes the metric repeatable before/after retrieval improvements.
TEST_CASES = [
    {
        "id": "identity_name",
        "question": "Tên của ứng viên trong CV là gì?",
        "expected_evidence": "An Pham Hoang",
        "reference_chunk": 1,
    },
    {
        "id": "identity_role",
        "question": "Chức danh hiện trên đầu CV là gì?",
        "expected_evidence": "Robotics AI Engineer",
        "reference_chunk": 1,
    },
    {
        "id": "education_bachelor_gpa",
        "question": "GPA bậc cử nhân của ứng viên là bao nhiêu?",
        "expected_evidence": "GPA 7.69",
        "reference_chunk": 1,
    },
    {
        "id": "education_master_school",
        "question": "Ứng viên đang học thạc sĩ tại trường nào?",
        "expected_evidence": "Iu Internationale Hochschule",
        "reference_chunk": 1,
    },
    {
        "id": "experience_renesas",
        "question": "Ứng viên làm việc tại Renesas Design Vietnam trong thời gian nào?",
        "expected_evidence": "Renesas Design Vietnam 5-2023 - 7-2024",
        "reference_chunk": 3,
    },
    {
        "id": "experience_cvedix",
        "question": "Tại CVEDIX, ứng viên làm vị trí nào?",
        "expected_evidence": "CVEDIX- Intern AI engineer",
        "reference_chunk": 4,
    },
    {
        "id": "project_master_thesis",
        "question": "Tên đề tài master thesis của ứng viên là gì?",
        "expected_evidence": "From Video to Simulation: A Real2Sim Pipeline",
        "reference_chunk": 5,
    },
    {
        "id": "project_bachelor_thesis",
        "question": "Bachelor thesis của ứng viên nghiên cứu về chủ đề gì?",
        "expected_evidence": "Tele-Manipulator",
        "reference_chunk": 6,
    },
    {
        "id": "project_azure_rag",
        "question": "Dự án chatbot Azure của ứng viên có đường dẫn GitHub nào?",
        "expected_evidence": "azure-rag_1",
        "reference_chunk": 10,
    },
    {
        "id": "certification_ielts",
        "question": "Điểm IELTS được ghi trong CV là bao nhiêu?",
        "expected_evidence": "IELTS: 6.5",
        "reference_chunk": 11,
    },
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def evaluate(source_file: str, chroma_dir: Path, k: int) -> dict:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=str(chroma_dir), embedding_function=embeddings
    )

    cases = []
    for case in TEST_CASES:
        documents = vectorstore.similarity_search(
            case["question"], k=k, filter={"source_file": source_file}
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

    hit_count = sum(case["hit"] for case in cases)
    return {
        "metric": f"retrieval_hit_at_{k}",
        "definition": (
            f"A test case is a hit when at least one of the top {k} retrieved chunks "
            "contains its predefined expected evidence text."
        ),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "source_file": source_file,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(cases),
        "hits": hit_count,
        "accuracy": hit_count / len(cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure CV retrieval Hit@k.")
    parser.add_argument("--source-file", default=DEFAULT_SOURCE_FILE)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")
    if not args.chroma_dir.exists():
        parser.error(f"Chroma directory does not exist: {args.chroma_dir}")

    result = evaluate(args.source_file, args.chroma_dir, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Retrieval Hit@{args.k}: {result['hits']}/{result['total_cases']} "
        f"({result['accuracy']:.0%})"
    )
    print(f"Saved baseline to {args.output}")


if __name__ == "__main__":
    main()
