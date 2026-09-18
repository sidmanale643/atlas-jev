import argparse
from datetime import datetime

from atlas_jev.pipeline import IngestReport, MemoryPipeline
from atlas_jev.store import Memory, MemoryEvent


def main() -> None:
    parser = argparse.ArgumentParser(prog="atlas-jev", description="Jev-gated memory store")
    sub = parser.add_subparsers(dest="command", required=True)

    add_parser = sub.add_parser("add", help="Ingest text: extract, gate, and store memories")
    add_parser.add_argument("text", help="Text to extract memories from")

    search_parser = sub.add_parser(
        "search", help="Hybrid search, then drop memories Jev judges unrelated"
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5)

    sub.add_parser("list", help="List all stored memories")

    history_parser = sub.add_parser("history", help="Show gate decisions and memory writes")
    history_parser.add_argument("memory_id", nargs="?", help="Optional memory id or prefix")

    revert_parser = sub.add_parser("revert", help="Restore a memory's previous value")
    revert_parser.add_argument("memory_id", help="Memory id or unique prefix")

    args = parser.parse_args()
    pipeline = MemoryPipeline()

    match args.command:
        case "add":
            _print_ingest(pipeline.add(args.text))
        case "search":
            hits = pipeline.search(args.query, limit=args.limit)
            if not hits:
                print("No memories found.")
            for hit in hits:
                print(f"[{hit.score:.3f} rel={hit.relevance:.2f}] {_format_memory(hit.memory)}")
        case "list":
            memories = pipeline.list_memories()
            if not memories:
                print("No memories stored.")
            for memory in memories:
                print(_format_memory(memory))
        case "history":
            events = pipeline.history(args.memory_id)
            if not events:
                print("No decision history.")
            for event in events:
                print(_format_event(event))
        case "revert":
            memory = pipeline.revert(args.memory_id)
            print(_format_memory(memory))
        case _:
            raise AssertionError(f"unhandled command: {args.command}")


def _print_ingest(report: IngestReport) -> None:
    print(f"extracted={_iso(report.extracted_at)}")
    print(f"source: {report.source_text}")
    if not report.results:
        print("No memories extracted.")
        return
    for result in report.results:
        target = f" target={result.decision.target_id[:8]}" if result.decision.target_id else ""
        print(
            f"[{result.action_taken:8s}] ({result.candidate.type:12s}) {result.candidate.text} "
            f"(extract_conf={result.candidate.confidence:.2f}, worth={result.decision.worth:.2f}, "
            f"op={result.decision.operation}, op_conf={result.decision.operation_confidence:.2f}"
            f"{target} conflict={result.decision.conflict:.2f})"
        )


def _format_memory(memory: Memory) -> str:
    previous = f' prev="{memory.previous_text}"' if memory.previous_text else ""
    target = f" target={memory.target_id[:8]}" if memory.target_id else ""
    return (
        f"{memory.id[:8]}  ({memory.type:12s}) {memory.text}\n"
        f"  extracted={_iso(memory.extracted_at)} extract_conf={memory.extraction_confidence:.2f} "
        f"worth={memory.confidence:.2f} "
        f"op={memory.operation or '-'} op_conf={memory.operation_confidence:.2f}"
        f"{target} conflict={memory.conflict:.2f}{previous}\n"
        f"  source: {memory.source_text}"
    )


def _format_event(event: MemoryEvent) -> str:
    memory_id = (event.memory_id or "-")[:8]
    previous = f' prev="{event.previous_text}"' if event.previous_text else ""
    target = f" target={event.target_id[:8]}" if event.target_id else ""
    return (
        f"{_iso(event.created_at)}  [{event.action_taken:8s}] ({event.candidate_type:12s}) "
        f"{event.candidate_text}\n"
        f"  extracted={_iso(event.extracted_at)} memory={memory_id} "
        f"extract_conf={event.extraction_confidence:.2f} worth={event.worth:.2f} "
        f"op={event.operation} op_conf={event.operation_confidence:.2f}"
        f"{target} conflict={event.conflict:.2f}{previous}\n"
        f"  source: {event.source_text}"
    )


def _iso(timestamp: float) -> str:
    if not timestamp:
        return "-"
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
