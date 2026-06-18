from __future__ import annotations

import asyncio
import hashlib
import logging
import os

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Final

from aiogram import Bot, Dispatcher, F, types
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatAction, ChatType
from aiogram.filters import CommandStart
from aiogram.utils.chat_action import ChatActionSender
from dotenv import load_dotenv

from agent import analyze_request
from config.constants import settings
from database import fetch_order_by_number, fetch_orders_summary, init_db, seed_database
from intent_client import IntentClient, build_default_intent_client
from ml_pipeline.rag_service import StorePolicyRAG, should_route_to_rag
from schemas import CustomerRequestAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PROJECT_ROOT: Final[Path] = Path(__file__).parent
_POLICY_PDF: Final[Path] = _PROJECT_ROOT / "documents" / "refund_policy.pdf"
_MARKDOWN_SPECIALS: Final[frozenset[str]] = frozenset("\\`*_{}[]()#+-.!|>$=~")

load_dotenv(_PROJECT_ROOT / ".env")

LOCAL_PROXY: Final[str | None] = os.getenv("HTTP_PROXY") or None
MANAGER_CHAT_ID: Final[str | None] = os.getenv("MANAGER_CHAT_ID") or None

if not MANAGER_CHAT_ID:
    logger.warning(
        "MANAGER_CHAT_ID is not set; escalation alerts will be logged only, "
        "not delivered to a human manager.",
    )

session = AiohttpSession(proxy=LOCAL_PROXY)
bot = Bot(token=settings.bot.TOKEN, session=session)

dp = Dispatcher()

intent_client: IntentClient = build_default_intent_client()


@dataclass
class RagServiceState:
    value: StorePolicyRAG | None = None


rag_service = RagServiceState()


class NativeRichDraft:
    def __init__(self, message: types.Message, draft_key: str) -> None:
        self._message = message
        self._draft_id = _build_draft_id(message, draft_key)
        self._enabled = _can_stream_rich_draft(message)
        self._last_text = ""

    async def update_html(self, text: str) -> None:
        if not self._enabled or not text or text == self._last_text:
            return

        self._last_text = text
        try:
            await bot.send_rich_message_draft(
                chat_id=self._message.chat.id,
                draft_id=self._draft_id,
                rich_message=_input_rich_html_text(text),
                **_thread_kwargs(self._message),
            )
        except Exception:
            self._enabled = False
            logger.exception(
                "Native rich draft streaming failed for chat_id=%s",
                self._message.chat.id,
            )


@dp.message(CommandStart())
async def cmd_start(message: types.Message) -> None:
    await message.answer(
        "Hello! I'm your virtual customer-support manager.\n\n"
        "Tell me what's going on with your order, ask a question, or share "
        "feedback. I'll do my best to help you right away.\n\n"
        "Tip: if you mention your order number (for example, <code>ORD-1001</code>), "
        "address, or phone, I'll pick them up automatically.",
        parse_mode="HTML",
    )


def _can_stream_rich_draft(message: types.Message) -> bool:
    return message.chat.type in (ChatType.PRIVATE, "private") and isinstance(message.chat.id, int)


def _thread_kwargs(message: types.Message) -> dict[str, int]:
    message_thread_id = getattr(message, "message_thread_id", None)
    if message_thread_id is None:
        return {}
    return {"message_thread_id": message_thread_id}


def _build_draft_id(message: types.Message, draft_key: str) -> int:
    raw = f"{message.chat.id}:{message.message_id}:{draft_key}".encode()
    draft_id = int.from_bytes(hashlib.blake2s(raw, digest_size=4).digest(), "big") & 0x7FFFFFFF
    return draft_id or 1


def _input_rich_html_text(text: str) -> types.InputRichMessage:
    return types.InputRichMessage(html=escape(text), skip_entity_detection=True)


def _input_rich_html(html: str) -> types.InputRichMessage:
    return types.InputRichMessage(html=html)


def _input_rich_markdown(markdown: str) -> types.InputRichMessage:
    return types.InputRichMessage(markdown=markdown, skip_entity_detection=True)


async def _send_rich_html_text(message: types.Message, text: str) -> None:
    await bot.send_rich_message(
        chat_id=message.chat.id,
        rich_message=_input_rich_html_text(text),
        **_thread_kwargs(message),
    )


async def _send_rich_html(message: types.Message, html: str) -> None:
    await bot.send_rich_message(
        chat_id=message.chat.id,
        rich_message=_input_rich_html(html),
        **_thread_kwargs(message),
    )


async def _send_manager_markdown(markdown: str) -> None:
    if not MANAGER_CHAT_ID:
        return
    try:
        await bot.send_rich_message(
            chat_id=MANAGER_CHAT_ID,
            rich_message=_input_rich_markdown(markdown),
        )
    except Exception:
        logger.exception("Failed to deliver markdown table to MANAGER_CHAT_ID=%s", MANAGER_CHAT_ID)


def _extract_classifier_intent(classification: dict | None) -> str | None:
    if not classification:
        return None
    raw = classification.get("intents")
    if raw is None:
        return None
    if isinstance(raw, list):
        return ", ".join(str(item) for item in raw) if raw else None
    return str(raw)


def _format_value(value: str | None) -> str:
    if value is None or value == "":
        return "<i>N/A</i>"
    return f"<code>{escape(value)}</code>"


def _escape_markdown_text(value: object) -> str:
    return "".join(f"\\{char}" if char in _MARKDOWN_SPECIALS else char for char in str(value))


def _escape_markdown_table_cell(value: object) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return _escape_markdown_text(text)


def _format_table_value(value: object | None) -> str:
    if value is None or value == "":
        return "N/A"
    return _escape_markdown_table_cell(value)


def _build_markdown_table(title: str, rows: list[tuple[str, object | None]], footer: str = "") -> str:
    lines = [
        title,
        "",
        "| Field | Value |",
        "|:------|:------|",
    ]
    lines.extend(
        f"| {_escape_markdown_table_cell(field)} | {_format_table_value(value)} |"
        for field, value in rows
    )
    if footer:
        lines.append("")
        lines.append(footer)
    return "\n".join(lines)


async def _lookup_order(order_number: str) -> dict[str, Any] | None:
    try:
        return await fetch_order_by_number(order_number)
    except Exception:
        logger.exception("DB lookup for order '%s' failed", order_number)
        return None


async def _lookup_summary() -> dict[str, Any] | None:
    try:
        return await fetch_orders_summary()
    except Exception:
        logger.exception("DB summary lookup failed")
        return None


def _build_db_facts_block(
    analysis: CustomerRequestAnalysis,
    *,
    db_order: dict[str, Any] | None,
    orders_summary: dict[str, Any] | None,
) -> str | None:
    sections: list[str] = []

    if analysis.order_number:
        order_number_html = escape(analysis.order_number)
        if db_order is None:
            sections.append(
                f"Order <code>{order_number_html}</code> was <b>not found</b> "
                f"in our system. Please double-check the number.",
            )
        else:
            status = escape(str(db_order.get("status", "unknown")))
            sections.append(
                f"Order <code>{order_number_html}</code> status: "
                f"<b>{status}</b>",
            )

    if analysis.wants_overall_summary and orders_summary is not None:
        total = int(orders_summary.get("total_orders", 0))
        by_status: dict[str, int] = dict(orders_summary.get("by_status", {}))
        if total == 0:
            sections.append("There are no orders in our system yet.")
        else:
            summary_lines = [f"Total orders in system: <b>{total}</b>"]
            summary_lines.extend(
                f"{escape(status)}: <b>{count}</b>" for status, count in by_status.items()
            )
            sections.append("\n".join(summary_lines))
    elif analysis.wants_overall_summary and orders_summary is None:
        sections.append(
            "Overall statistics are temporarily unavailable, sorry about that.",
        )

    if not sections:
        return None

    return "\n\n".join(sections)


def _build_extracted_card(
    user: types.User,
    user_text: str,
    analysis: CustomerRequestAnalysis,
    ai_response_text: str,
    *,
    db_order: dict[str, Any] | None = None,
    classifier_intent: str | None = None,
) -> str:
    rows: list[tuple[str, object | None]] = []

    if classifier_intent is not None:
        rows.append(("Intent", classifier_intent))
        # rows.append(("Intent (Agent)", analysis.intent))
    else:
        rows.append(("Intent", analysis.intent))

    rows.extend(
        [
            ("Order", analysis.order_number),
        ],
    )

    if analysis.order_number:
        rows.append(
            (
                "Order Status",
                "not found in our system" if db_order is None else db_order.get("status"),
            ),
        )

    rows.extend(
        [
            ("Sentiment", analysis.sentiment),
        ],
    )

    mention = f"[{_escape_markdown_text(user.full_name)}](tg://user?id={user.id})"
    if user.username:
        mention = f"[@{_escape_markdown_text(user.username)}](https://t.me/{user.username})"

    title = f"# {mention}:\n## {_escape_markdown_text(user_text)}"
    footer = f"> \n> {_escape_markdown_text(ai_response_text).replace('\n', '\n> ')}"

    return _build_markdown_table(title, rows, footer)


def _build_rag_card(
    user: types.User,
    user_text: str,
    classifier_intent: str | None,
    policy_answer: str,
    analysis: CustomerRequestAnalysis | None,
) -> str:
    mention = f"[{_escape_markdown_text(user.full_name)}](tg://user?id={user.id})"
    if user.username:
        mention = f"[@{_escape_markdown_text(user.username)}](https://t.me/{user.username})"

    title = f"# {mention}:\n## {_escape_markdown_text(user_text)}"
    footer = f"> \n> {_escape_markdown_text(policy_answer).replace('\n', '\n> ')}"

    order_val = analysis.order_number if (analysis and analysis.order_number) else "N/A"
    sentiment_val = analysis.sentiment if (analysis and analysis.sentiment) else "N/A"

    return _build_markdown_table(
        title,
        [
            ("Intent", classifier_intent),
            ("Order", order_val),
            ("Sentiment", sentiment_val),
            ("Source", "Store Policy RAG"),
        ],
        footer,
    )


def _build_manager_alert(
    *,
    user: types.User,
    user_text: str,
    analysis: CustomerRequestAnalysis,
) -> str:
    username = f"@{user.username}" if user.username else "(no username)"
    return (
        "<b>🤬URGENT: Angry Customer!🤬</b>\n"
        f"<b>User</b>: {escape(username)} (ID: <code>{user.id}</code>)"
        # f"<b>Intent</b>: {escape(analysis.intent)}\n"
        # f"<b>Sentiment</b>: {escape(analysis.sentiment)}\n"
        # f"<b>Message</b>: {escape(user_text)}"
    )


async def _notify_manager(
    *,
    user: types.User,
    user_text: str,
    analysis: CustomerRequestAnalysis,
) -> None:
    if not MANAGER_CHAT_ID:
        return

    alert_text = _build_manager_alert(
        user=user,
        user_text=user_text,
        analysis=analysis,
    )

    try:
        await bot.send_message(
            chat_id=MANAGER_CHAT_ID,
            text=alert_text,
            parse_mode="HTML",
        )
        logger.info("Escalation alert delivered to manager (user_id=%s)", user.id)
    except Exception:
        logger.exception(
            "Failed to deliver escalation alert to MANAGER_CHAT_ID=%s",
            MANAGER_CHAT_ID,
        )


async def _stream_rag_policy_answer(message: types.Message, user_text: str) -> str:
    service = rag_service.value
    if service is None:
        return ""

    draft = NativeRichDraft(message, "rag")
    policy_answer = ""

    async for partial_answer in service.ask_stream(user_text):
        if partial_answer:
            policy_answer = partial_answer
            await draft.update_html(partial_answer)

    if not policy_answer:
        policy_answer = await asyncio.to_thread(service.ask, user_text)

    return policy_answer


@dp.message(F.text)
async def handle_text(message: types.Message) -> None:
    user_text = (message.text or "").strip()
    if not user_text:
        await message.answer("Please send a non-empty text message.")
        return

    if message.from_user is None:
        await message.answer("Sorry, I couldn't identify the sender. Please try again.")
        return

    async with ChatActionSender(
        bot=bot,
        chat_id=message.chat.id,
        action=ChatAction.TYPING,
    ):
        classification = await intent_client.classify(user_text)

    intent_hint = _extract_classifier_intent(classification)

    if should_route_to_rag(classification) and rag_service.value is not None:
        async with ChatActionSender(
            bot=bot,
            chat_id=message.chat.id,
            action=ChatAction.TYPING,
        ):
            analysis = await analyze_request(user_text, intent_hint=intent_hint)

        try:
            policy_answer = await _stream_rag_policy_answer(message, user_text)
        except Exception:
            logger.exception("RAG policy lookup failed for user_id=%s", message.from_user.id)
            await message.answer(
                "Sorry, I couldn't look up the store policy right now. "
                "Please try again in a moment.",
            )
            return

        try:
            await _send_rich_html_text(message, policy_answer)
        except Exception:
            logger.exception("Failed to deliver RAG reply to user_id=%s", message.from_user.id)
            return

        await _send_manager_markdown(
            _build_rag_card(message.from_user, user_text, intent_hint, policy_answer, analysis)
        )

        if analysis is not None and analysis.sentiment == "Critical/Angry":
            await _notify_manager(
                user=message.from_user,
                user_text=user_text,
                analysis=analysis,
            )
        return

    draft = NativeRichDraft(message, "analysis")

    async def stream_ai_response(partial_text: str) -> None:
        await draft.update_html(partial_text)

    async with ChatActionSender(
        bot=bot,
        chat_id=message.chat.id,
        action=ChatAction.TYPING,
    ):
        analysis = await analyze_request(
            user_text,
            intent_hint=intent_hint,
            on_ai_response=stream_ai_response,
        )

        db_order: dict[str, Any] | None = None
        orders_summary: dict[str, Any] | None = None
        if analysis is not None:
            if analysis.order_number:
                db_order = await _lookup_order(analysis.order_number)
            if analysis.wants_overall_summary:
                orders_summary = await _lookup_summary()

    if analysis is None:
        await message.answer(
            "Sorry, I couldn't process your request right now. "
            "Please try again in a moment.",
        )
        return

    facts_block = _build_db_facts_block(
        analysis,
        db_order=db_order,
        orders_summary=orders_summary,
    )

    try:
        await _send_rich_html_text(message, analysis.ai_response_text)
        if facts_block is not None:
            await _send_rich_html(message, facts_block)
    except Exception:
        logger.exception("Failed to deliver reply to user_id=%s", message.from_user.id)
        return

    await _send_manager_markdown(
        _build_extracted_card(
            message.from_user,
            user_text,
            analysis,
            analysis.ai_response_text,
            db_order=db_order,
            classifier_intent=intent_hint,
        )
    )

    if analysis.sentiment == "Critical/Angry":
        await _notify_manager(
            user=message.from_user,
            user_text=user_text,
            analysis=analysis,
        )


async def _init_rag() -> None:
    try:
        service = StorePolicyRAG()
        await asyncio.to_thread(service.build_or_load_index, str(_POLICY_PDF))
        rag_service.value = service
        logger.info("Store policy RAG index is ready")
    except Exception:
        logger.exception("Failed to initialize StorePolicyRAG; RAG branch disabled")
        rag_service.value = None


async def _on_startup() -> None:
    await init_db()
    await seed_database()
    await _init_rag()


async def _on_shutdown() -> None:
    await intent_client.close()
    await bot.session.close()


async def main() -> None:
    await _on_startup()
    logger.info("Bot is up, starting polling...")
    try:
        await dp.start_polling(bot, polling_timeout=5)
    finally:
        await _on_shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
