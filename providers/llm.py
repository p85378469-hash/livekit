# ==============================================================================
# providers/llm.py — The AI Brain (OpenRouter / Groq Streaming LLM)
#
# IN SIMPLE WORDS:
# This file is the "Brain" of the voice agent.
# When the customer asks a question or says something unpredictable, this file sends
# the conversation history to the AI (OpenRouter or Groq) to generate a response in Telugu.
#
# Key Features:
# 1. Streaming: Words are received word-by-word, not waiting for the full answer!
# 2. Sentence Chunking: Words are grouped into complete sentences so the voice
#    generator (TTS) sounds natural and expressive.
# 3. Interruption Safety: If the caller interrupts while the AI is generating words,
#    the loop stops immediately and throws away remaining text.
# ==============================================================================

from __future__ import annotations

import os
import time
from typing import AsyncIterator
import structlog
from openai import AsyncOpenAI

from telemetry.latency import LatencyTracker, Stage

log = structlog.get_logger(__name__)

# Default model used for OpenRouter (Llama 3.1 70B has great multilingual and Telugu abilities)
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.1-70b-instruct"
DEFAULT_GROQ_MODEL = "llama-3.1-70b-versatile"


class GroqLLM:
    """
    Streaming AI wrapper supporting OpenRouter (or Groq) via the OpenAI-compatible API.
    """

    def __init__(self, latency: LatencyTracker) -> None:
        """
        Reads OPENROUTER_API_KEY (or GROQ_API_KEY) from environment and initializes client.
        """
        self._latency = latency

        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")

        if openrouter_key:
            # Use OpenRouter endpoint
            self.provider = "openrouter"
            self.model = os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
            self._client = AsyncOpenAI(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://livekit.io",
                    "X-Title": "Telugu Voice Agent",
                },
            )
        elif groq_key:
            # Fallback to Groq if user uses Groq
            self.provider = "groq"
            self.model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
            self._client = AsyncOpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
            )
        else:
            raise EnvironmentError("Neither OPENROUTER_API_KEY nor GROQ_API_KEY is set in environment")

        log.info("llm.initialized", provider=self.provider, model=self.model)

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        turn_id: int,
        is_turn_current_fn,
    ) -> AsyncIterator[str]:
        """
        Streams words from the AI and yields complete sentence chunks.
        
        messages: The chat history and system instructions.
        turn_id: Current turn number.
        is_turn_current_fn: Function that checks if caller interrupted.
        """
        log.info("llm.generation_start", turn_id=turn_id, provider=self.provider, model=self.model)
        first_token_recorded = False
        accumulated = ""

        try:
            # Send prompt with streaming enabled
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,        # Keep answers short and sweet for phone calls
                temperature=0.3,       # Low creativity for reliable Telugu facts
                stream=True,
            )

            async for chunk in stream:
                # Safety check: If user started talking, abort AI generation immediately!
                if not is_turn_current_fn():
                    log.warning(
                        "llm.stale_turn_discarded",
                        turn_id=turn_id,
                        accumulated_so_far=accumulated[:60],
                    )
                    return

                if not chunk.choices or not chunk.choices[0].delta:
                    continue

                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue

                # Record T3: The exact millisecond the very first token arrived from AI
                if not first_token_recorded:
                    self._latency.record(Stage.LLM_FIRST_TOKEN, time.time() * 1000)
                    first_token_recorded = True
                    log.debug("llm.first_token", turn_id=turn_id)

                accumulated += delta

                # Wait until we have a full sentence (ends with ., ?, !, or Telugu purna viram ।)
                sentence_chunk = self._extract_sentence_chunk(accumulated)
                if sentence_chunk:
                    accumulated = accumulated[len(sentence_chunk):]
                    log.debug(
                        "llm.chunk_yielded",
                        chunk_preview=sentence_chunk[:40],
                        turn_id=turn_id,
                    )
                    # Yield complete sentence to Cartesia TTS
                    yield sentence_chunk

            # Send any leftover words at the end
            if accumulated.strip() and is_turn_current_fn():
                yield accumulated.strip()

        except Exception as e:
            log.error("llm.generation_error", error=str(e), turn_id=turn_id)
            raise

        log.info("llm.generation_complete", turn_id=turn_id)

    def _extract_sentence_chunk(self, text: str) -> str:
        """
        Pulls out the first completed sentence from accumulated words.
        Text-to-Speech models sound much more human when given complete sentences.
        """
        for i, char in enumerate(text):
            if char in (".", "?", "!", "।", "\n"):
                return text[: i + 1].strip()
        return ""
