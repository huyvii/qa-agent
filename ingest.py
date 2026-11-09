"""Index one PDF into the configured local Chroma collection and SQLite registry."""

import argparse
from pathlib import Path

import chat_db
from rag_core import compute_file_sha256, delete_document_vectors, ingest_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into the QA Agent vector store.")
    parser.add_argument("pdf", type=Path, help="path to the PDF file")
    parser.add_argument(
        "--source-file",
        help="stable source_file metadata value (defaults to the PDF filename)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-index even if the same file content hash is already recorded",
    )
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error(f"PDF file not found: {args.pdf}")

    file_name = args.source_file or args.pdf.name
    chat_db.init_db()
    content_hash = compute_file_sha256(args.pdf)
    existing = chat_db.get_document(file_name)

    if not args.force and existing and existing.get("content_hash") == content_hash:
        print(
            f"Skipped '{file_name}': identical SHA-256 already indexed "
            f"({existing['chunk_count']} chunks). Pass --force to re-index."
        )
        return

    timings = {}
    try:
        result = ingest_pdf(str(args.pdf), source_file=file_name, timings=timings)
    except Exception as exc:
        if existing and existing.get("content_hash"):
            try:
                delete_document_vectors(file_name, keep_content_hash=existing["content_hash"])
            except Exception:
                pass
        parser.exit(1, f"PDF ingest failed ({type(exc).__name__}). Check the file and logs.\n")

    delete_document_vectors(file_name, keep_content_hash=result["content_hash"])
    chat_db.save_document(file_name, result["content_hash"], result["chunks"])

    print(f"Indexed {result['chunks']} chunks from {file_name}")
    if result["empty_pages"]:
        print(f"Pages with little or no text (one-based): {[page + 1 for page in result['empty_pages']]}")
    print(f"Completed in {timings['total_seconds']:.2f}s")


if __name__ == "__main__":
    main()
