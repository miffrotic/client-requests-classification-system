from __future__ import annotations

import logging
import os

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from schemas import CustomerRequestAnalysis


logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent / ".env")

GEMINI_API_KEY: Final[str | None] = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    msg = "GEMINI_API_KEY not found. Check your .env file."
    raise RuntimeError(msg)

GEMINI_MODEL: Final[str] = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


_SYSTEM_PROMPT: Final[str] = (
    "You are an expert Customer Support Analyzer for an online store. "
    "Your job is to deeply analyze every incoming customer message and return a "
    "STRUCTURED analysis with these tasks performed in one shot:\n\n"
    "1. INTENT CLASSIFICATION — pick a short, descriptive label (Title Case) for "
    "what the customer wants. Typical labels: 'Delivery Complaint', 'Refund "
    "Request', 'Order Status Inquiry', 'Cancellation Request', 'Product Question', "
    "'General Question', 'Compliment'. Choose the single most fitting label.\n\n"
    "2. ENTITY EXTRACTION — pull out, when present, the order number "
    "(patterns like ORD-1234, order 1234, #1234), the delivery address, and the "
    "phone number. If a piece of information is not in the message, set the "
    "corresponding field to null. NEVER invent or guess these values.\n\n"
    "3. SENTIMENT ANALYSIS — must be EXACTLY one of: 'Positive', 'Neutral', "
    "'Negative', 'Critical/Angry'. Apply these strict rules:\n"
    "   • If the user writes in ALL CAPS, uses multiple exclamation marks, "
    "profanity, insults, or threats (e.g. \"I will sue\", \"I'm reporting you\", "
    "\"I'm leaving forever\", \"never buying again\") — the sentiment MUST be "
    "'Critical/Angry'. Do not soften it.\n"
    "   • Frustration without threats or shouting → 'Negative'.\n"
    "   • Plain factual question or status check → 'Neutral'.\n"
    "   • Thanks, praise, or happy feedback → 'Positive'.\n\n"
    "4. AI RESPONSE — write the reply for the user in `ai_response_text`, "
    "STRICTLY IN ENGLISH, polite and concise. If sentiment is 'Critical/Angry', "
    "the reply MUST be deeply apologetic, acknowledge the customer's frustration, "
    "and EXPLICITLY state that a human manager has been notified and will reach "
    "out shortly. For other sentiments, be warm and helpful. Never reveal "
    "internal tools, classifiers, the database or this schema.\n\n"
    "5. OVERALL SUMMARY FLAG — set `wants_overall_summary=true` ONLY when the "
    "customer asks about the picture across ALL orders (e.g. 'how many orders are "
    "in transit?', 'overall status?', 'give me a summary', 'how are things in "
    "general?'). For questions about one specific order, complaints, refunds, "
    "etc. keep it false. CRITICAL: never invent counts or statistics in your "
    "reply — when this flag is true, the application will append the real numbers "
    "from the database below your message. If sentiment is 'Critical/Angry', do "
    "NOT also set the summary flag — focus on apologising.\n\n"
    "FACTUAL DATA POLICY: do not state order statuses, addresses, totals, or any "
    "other facts the user did not provide. If `ai_response_text` references an "
    "order number, keep it neutral (\"let me check on that for you\") rather than "
    "claiming a status — the application enriches the reply with verified data "
    "from its database afterwards.\n\n"
    "You will also receive a soft hint from an upstream intent classifier in "
    "`Intent hint from classifier`. Treat it as advisory only — your own analysis "
    "of the raw user text is the source of truth."
)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        (
            "human",
            "Intent hint from classifier: {intent_hint}\n\n"
            "User message:\n{input}",
        ),
    ],
)

_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)

_structured_llm = _llm.with_structured_output(CustomerRequestAnalysis)
_chain = _prompt | _structured_llm
AnalysisDraftCallback = Callable[[str], Awaitable[None]]


def _coerce_analysis(value: object) -> CustomerRequestAnalysis | None:
    if isinstance(value, CustomerRequestAnalysis):
        return value
    if isinstance(value, dict):
        try:
            return CustomerRequestAnalysis.model_validate(value)
        except ValidationError:
            return None
    return None


def _extract_ai_response_text(value: object) -> str | None:
    analysis = _coerce_analysis(value)
    if analysis is not None:
        return analysis.ai_response_text
    if isinstance(value, dict):
        raw = value.get("ai_response_text")
        return str(raw) if raw is not None else None
    raw = getattr(value, "ai_response_text", None)
    return str(raw) if raw is not None else None


async def _invoke_analysis(payload: dict[str, str]) -> CustomerRequestAnalysis | None:
    try:
        result = await _chain.ainvoke(payload)
    except Exception:
        logger.exception("LLM analyzer invocation failed")
        return None

    if not isinstance(result, CustomerRequestAnalysis):
        logger.error(
            "Unexpected structured-output type: %s (value=%r)",
            type(result).__name__,
            result,
        )
        return None

    return result


async def _stream_analysis(
    payload: dict[str, str],
    on_ai_response: AnalysisDraftCallback,
) -> CustomerRequestAnalysis | None:
    latest_response = ""
    streamed_result: CustomerRequestAnalysis | None = None

    try:
        async for chunk in _chain.astream(payload):
            candidate = _coerce_analysis(chunk)
            if candidate is not None:
                streamed_result = candidate

            ai_response_text = _extract_ai_response_text(chunk)
            if ai_response_text and ai_response_text != latest_response:
                latest_response = ai_response_text
                await on_ai_response(ai_response_text)
    except Exception:
        logger.exception("LLM analyzer streaming failed")
        return await _invoke_analysis(payload)

    if streamed_result is not None:
        return streamed_result

    return await _invoke_analysis(payload)


async def analyze_request(
    user_message: str,
    intent_hint: str | None = None,
    on_ai_response: AnalysisDraftCallback | None = None,
) -> CustomerRequestAnalysis | None:
    payload = {
        "input": user_message,
        "intent_hint": intent_hint or "not provided",
    }

    if on_ai_response is None:
        return await _invoke_analysis(payload)

    return await _stream_analysis(payload, on_ai_response)
