#!/usr/bin/env python3
"""Measure Groq GPT-OSS prompt-cache usage and streaming latency.

Run:
    python scripts/test_groq_prompt_cache.py
    python scripts/test_groq_prompt_cache.py --runs 5
    python scripts/test_groq_prompt_cache.py --synthetic-prefix

The default mode tests the real stable voice-agent system prompt. The optional
synthetic mode checks that cache reporting works with a deliberately long,
unchanging prefix; it is not intended for production prompts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from pipeline.context import SYSTEM_PROMPT

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"


def make_synthetic_prefix() -> str:
    notes = [
        f"Stable cache probe reference {i:03d}: This fixed sentence is included only "
        "to create a long, identical prefix for verifying provider cache accounting."
        for i in range(240)
    ]
    return SYSTEM_PROMPT + "\n\nCACHE PROBE ONLY — NOT PRODUCTION INSTRUCTIONS\n" + "\n".join(notes)


async def run_trial(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    trial: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_output_ms: float | None = None
    usage: Any = None

    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Trial {trial}: Reply briefly in Telugu. Say hello."},
        ],
        temperature=0.6,
        max_completion_tokens=96,
        reasoning_effort="low",
        stream=True,
        stream_options={"include_usage": True},
    )

    async with stream:
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.choices:
                delta = chunk.choices[0].delta
                visible = getattr(delta, "content", None)
                reasoning = getattr(delta, "reasoning", None)
                if first_output_ms is None and (visible or reasoning):
                    first_output_ms = (time.perf_counter() - started) * 1000

    wall_ms = (time.perf_counter() - started) * 1000
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    cached_tokens = getattr(details, "cached_tokens", 0) if details else 0

    return {
        "trial": trial,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cache_hit_rate_percent": round(cached_tokens / prompt_tokens * 100, 2) if prompt_tokens else None,
        "completion_tokens": completion_tokens,
        "first_output_ms": round(first_output_ms, 1) if first_output_ms is not None else None,
        "wall_latency_ms": round(wall_ms, 1),
        "provider_prompt_time_s": getattr(usage, "prompt_time", None) if usage else None,
        "provider_completion_time_s": getattr(usage, "completion_time", None) if usage else None,
        "provider_total_time_s": getattr(usage, "total_time", None) if usage else None,
        "usage_reported": usage is not None,
    }


async def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="Number of sequential requests (default: 3)")
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--synthetic-prefix",
        action="store_true",
        help="Use a long test-only prefix to diagnose cache eligibility",
    )
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GROQ_API_KEY is missing. Add it to your local .env file; do not commit it.")

    if args.runs < 2:
        raise SystemExit("--runs must be at least 2 to compare the first request with later requests.")

    prompt = make_synthetic_prefix() if args.synthetic_prefix else SYSTEM_PROMPT
    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    print(json.dumps({
        "test": "synthetic-prefix" if args.synthetic_prefix else "production-system-prompt",
        "model": args.model,
        "system_prompt_characters": len(prompt),
        "runs": args.runs,
    }, ensure_ascii=False))

    results = []
    try:
        for trial in range(1, args.runs + 1):
            result = await run_trial(client, args.model, prompt, trial)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
    finally:
        await client.close()

    total_prompt = sum(row["prompt_tokens"] or 0 for row in results)
    total_cached = sum(row["cached_tokens"] or 0 for row in results)
    print(json.dumps({
        "summary": True,
        "aggregate_prompt_tokens": total_prompt,
        "aggregate_cached_tokens": total_cached,
        "aggregate_cache_hit_rate_percent": round(total_cached / total_prompt * 100, 2) if total_prompt else None,
        "note": "Compare later trials with trial 1. A zero cache count is a cache miss, not proof that caching is disabled.",
    }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
