#!/usr/bin/env python3
"""Sync customer support RAG traces to Langfuse.

Usage:
  python scripts/sync_to_langfuse.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.telemetry.langfuse_client import get_langfuse_client, is_langfuse_configured

TRACES_FILE = PROJECT_ROOT / "traces.jsonl"


def sync_traces(dry_run: bool = False):
    settings = get_settings()
    print("=" * 80)
    print("LANGFUSE TRACE SYNC")
    print("=" * 80)
    print(f"Langfuse Host: {settings.langfuse_host}")
    print(f"Configured: {is_langfuse_configured()}")
    print(f"Dry Run: {dry_run}")
    print(f"Traces File: {TRACES_FILE}")

    if not TRACES_FILE.exists():
        print(f"Error: {TRACES_FILE} not found. Run python scripts/generate_traces.py first.")
        sys.exit(1)

    with open(TRACES_FILE, "r", encoding="utf-8") as f:
        traces = [json.loads(line) for line in f if line.strip()]

    print(f"Found {len(traces)} traces to sync.\n")

    if dry_run:
        print("[DRY RUN] Traces that would be uploaded:")
        for idx, t in enumerate(traces[:5], 1):
            print(f" [{idx:02d}] {t['trace_id']}: \"{t['user_query']}\" -> {t['model']}")
        print(f" ... and {len(traces) - 5} more traces.")
        print("\nDry run completed successfully. Set your LANGFUSE keys in .env to push live.")
        return

    client = get_langfuse_client()
    if not client:
        print("\n❌ Error: Langfuse is not configured.")
        print("Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in your .env file:")
        print("  LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("  LANGFUSE_SECRET_KEY=sk-lf-...")
        print("  LANGFUSE_HOST=https://cloud.langfuse.com")
        sys.exit(1)

    print("Uploading traces to Langfuse...")
    success_count = 0
    for idx, t in enumerate(traces, 1):
        try:
            # 1. Main Trace
            trace = client.trace(
                id=t.get("trace_id"),
                name="customer_support_rag_trace",
                input={"question": t.get("user_query")},
                output={"answer": t.get("raw_output")},
                metadata={
                    "prompt_version": t.get("prompt_template_version"),
                    "timestamp": t.get("timestamp"),
                    "open_coding_observation": t.get("open_coding_observation"),
                },
                tags=["week5-eval", "customer-support-rag", "seeded-pool"],
            )

            # 2. Retrieval Span
            chunks = t.get("retrieved_chunks", [])
            trace.span(
                name="vector_retrieval",
                input={"query": t.get("user_query")},
                output={"chunks": chunks},
                metadata={"top_k": len(chunks)},
            )

            # 3. Generation Span
            trace.generation(
                name="rag_generation",
                model=t.get("model", "claude-3-5-sonnet-20241022"),
                model_parameters=t.get("model_params", {"temperature": 0.0}),
                input=[
                    {"role": "system", "content": t.get("system_prompt", "")},
                    {"role": "user", "content": t.get("user_query", "")},
                ],
                output=t.get("raw_output", ""),
            )

            success_count += 1
            if idx % 10 == 0 or idx == len(traces):
                print(f" Synced {idx}/{len(traces)} traces...")
        except Exception as exc:
            print(f" Error uploading trace {t.get('trace_id')}: {exc}")

    # Flush all events to Langfuse server
    print("Flushing events to Langfuse...")
    client.flush()
    print(f"\n✅ Successfully synced {success_count}/{len(traces)} traces to Langfuse!")
    print(f"View your traces at: {settings.langfuse_host}")


def main():
    parser = argparse.ArgumentParser(description="Sync traces to Langfuse")
    parser.add_argument("--dry-run", action="store_true", help="Simulate upload without network calls")
    args = parser.parse_args()
    sync_traces(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
