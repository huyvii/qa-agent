"""Ask one question against the local vector store from the command line."""

import argparse

from rag_core import answer_question, create_llm, get_reranker, get_vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about indexed PDFs.")
    parser.add_argument("question", nargs="+", help="question to ask")
    parser.add_argument(
        "--source-file",
        help="limit retrieval to the original PDF filename used during ingest",
    )
    parser.add_argument("--k", type=int, default=3, help="number of context chunks")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="skip the cross-encoder reranker for faster CLI execution",
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    try:
        llm = create_llm()
        reranker = None if args.no_rerank else get_reranker()
        answer, sources = answer_question(
            " ".join(args.question),
            get_vectorstore(),
            llm,
            source_file_filter=args.source_file,
            k=args.k,
            reranker=reranker,
        )
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    except Exception as exc:
        parser.exit(1, f"Question failed ({type(exc).__name__}). Check configuration and logs.\n")

    print(answer)
    if sources:
        print("\nSources:")
        for source in sources:
            print(f"- {source}")


if __name__ == "__main__":
    main()
