"""Index one PDF into the configured local Chroma collection."""

import argparse
from pathlib import Path

from rag_core import ingest_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into the QA Agent vector store.")
    parser.add_argument("pdf", type=Path, help="path to the PDF file")
    parser.add_argument(
        "--source-file",
        help="stable source_file metadata value (defaults to the PDF filename)",
    )
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error(f"PDF file not found: {args.pdf}")

    timings = {}
    try:
        result = ingest_pdf(
            str(args.pdf), source_file=args.source_file or args.pdf.name, timings=timings
        )
    except Exception as exc:
        parser.exit(1, f"PDF ingest failed ({type(exc).__name__}). Check the file and logs.\n")

    print(f"Indexed {result['chunks']} chunks from {args.source_file or args.pdf.name}")
    if result["empty_pages"]:
        print(f"Pages with little or no text (one-based): {[page + 1 for page in result['empty_pages']]}")
    print(f"Completed in {timings['total_seconds']:.2f}s")


if __name__ == "__main__":
    main()
