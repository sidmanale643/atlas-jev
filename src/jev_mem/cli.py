import argparse

from jev_mem.pipeline import MemoryPipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="jev-mem", description="Jev-gated memory store")
    sub = parser.add_subparsers(dest="command", required=True)

    add_parser = sub.add_parser("add", help="Ingest text: extract, gate, and store memories")
    add_parser.add_argument("text", help="Text to extract memories from")

    search_parser = sub.add_parser("search", help="Semantic search over stored memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5)

    sub.add_parser("list", help="List all stored memories")

    args = parser.parse_args()
    pipeline = MemoryPipeline()

    if args.command == "add":
        report = pipeline.add(args.text)
        if not report.results:
            print("No memories extracted.")
        for result in report.results:
            print(
                f"[{result.action_taken:7s}] {result.candidate} "
                f"(worth={result.decision.worth:.2f}, op={result.decision.operation})"
            )
    elif args.command == "search":
        hits = pipeline.search(args.query, limit=args.limit)
        if not hits:
            print("No memories found.")
        for hit in hits:
            print(f"[{hit.distance:.3f}] {hit.memory.text}")
    elif args.command == "list":
        memories = pipeline.list_memories()
        if not memories:
            print("No memories stored.")
        for memory in memories:
            print(f"{memory.id[:8]}  {memory.text}")


if __name__ == "__main__":
    main()
