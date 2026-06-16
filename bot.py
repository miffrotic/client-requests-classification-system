from __future__ import annotations

import asyncio
import logging
import os
from html import escape
from pathlib import Path
from typing import Any, Final

from aiogram import Bot, Dispatcher, F, types
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.enums import ChatAction
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

load_dotenv(_PROJECT_ROOT / ".env")

# PROXY_URL: Final[str] = "https://tg-api-proxy.iasadulaev.workers.dev"
LOCAL_PROXY: Final[str] = os.getenv("HTTP_PROXY")
MANAGER_CHAT_ID: Final[str | None] = os.getenv("MANAGER_CHAT_ID") or None

if not MANAGER_CHAT_ID:
    logger.warning(
        "MANAGER_CHAT_ID is not set — escalation alerts will be logged only, "
        "not delivered to a human manager.",
    )

# custom_api = TelegramAPIServer.from_base(PROXY_URL)
# session = AiohttpSession(api=custom_api)
session = AiohttpSession(proxy=LOCAL_PROXY)
bot = Bot(token=settings.bot.TOKEN, session=session) # session=session

dp = Dispatcher()

intent_client: IntentClient = build_default_intent_client()
rag_service: StorePolicyRAG | None = None

@dp.message(CommandStart())
async def cmd_start(message: types.Message) -> None:
    """Greeting and a short usage hint."""
    await message.answer(
        "Hello! I'm your virtual customer-support manager.\n\n"
        "Tell me what's going on with your order, ask a question, or share "
        "feedback. I'll do my best to help you right away.\n\n"
        "Tip: if you mention your order number (for example, <code>ORD-1001</code>), "
        "address, or phone, I'll pick them up automatically.",
        parse_mode="HTML",
    )


def _extract_classifier_intent(classification: dict | None) -> str | None:
    """Pull the intent label out of the FastAPI classifier response, if any."""
    if not classification:
        return None
    raw = classification.get("intents")
    if raw is None:
        return None
    if isinstance(raw, list):
        return ", ".join(str(item) for item in raw) if raw else None
    return str(raw)


def _format_value(value: str | None) -> str:
    """HTML-safe representation of an optional extracted field."""
    if value is None or value == "":
        return "<i>—</i>"
    return f"<code>{escape(value)}</code>"


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
                f"• Order <code>{order_number_html}</code> was <b>not found</b> "
                f"in our system. Please double-check the number.",
            )
        else:
            status = escape(str(db_order.get("status", "unknown")))
            customer = escape(str(db_order.get("customer_name", "")))
            items = escape(str(db_order.get("items", "")))
            sections.append(
                f"• Order <code>{order_number_html}</code> — current status: "
                f"<b>{status}</b>"
                + (f"\n   Customer on file: {customer}" if customer else "")
                + (f"\n   Items: {items}" if items else ""),
            )

    if analysis.wants_overall_summary and orders_summary is not None:
        total = int(orders_summary.get("total_orders", 0))
        by_status: dict[str, int] = dict(orders_summary.get("by_status", {}))
        if total == 0:
            sections.append("• There are no orders in our system yet.")
        else:
            summary_lines = [f"• Total orders in system: <b>{total}</b>"]
            summary_lines.extend(
                f"   • {escape(status)}: <b>{count}</b>"
                for status, count in by_status.items()
            )
            sections.append("\n".join(summary_lines))
    elif analysis.wants_overall_summary and orders_summary is None:
        sections.append(
            "• Overall statistics are temporarily unavailable, sorry about that.",
        )

    if not sections:
        return None

    return "📦 <b>From Our System:</b>\n" + "\n".join(sections)


def _format_db_status(
    order_number: str | None,
    db_order: dict[str, Any] | None,
) -> str | None:
    """Render the «Order Status (from DB)» line, or ``None`` to omit it."""
    if not order_number:
        return None
    if db_order is None:
        return "• <b>Order Status (from DB)</b>: <i>not found in our system</i>"
    return f"• <b>Order Status (from DB)</b>: {_format_value(db_order.get('status'))}"


def _build_extracted_card(
    analysis: CustomerRequestAnalysis,
    *,
    db_order: dict[str, Any] | None = None,
    classifier_intent: str | None = None,
) -> str:
    lines = ["📊<b>Extracted Data:</b>"]

    if classifier_intent is not None:
        lines.append(
            f"• <b>Intent (Classifier)</b>: {_format_value(classifier_intent)}",
        )
        lines.append(
            f"• <b>Intent (Agent)</b>: {_format_value(analysis.intent)}",
        )
    else:
        lines.append(f"• <b>Intent</b>: {_format_value(analysis.intent)}")

    lines += [
        f"• <b>Order</b>: {_format_value(analysis.order_number)}",
        f"• <b>Address</b>: {_format_value(analysis.address)}",
        f"• <b>Phone</b>: {_format_value(analysis.phone)}",
        f"• <b>Sentiment</b>: {_format_value(analysis.sentiment)}",
    ]

    db_status_line = _format_db_status(analysis.order_number, db_order)
    if db_status_line is not None:
        lines.append(db_status_line)

    return "\n".join(lines)


def _build_rag_card(classifier_intent: str | None) -> str:
    lines = ["📊<b>Extracted Data:</b>"]
    lines.append(f"• <b>Intent (Classifier)</b>: {_format_value(classifier_intent)}")
    lines.append("• <b>Source</b>: <code>Store Policy RAG</code>")
    return "\n".join(lines)


def _build_manager_alert(
    *,
    user: types.User,
    user_text: str,
    analysis: CustomerRequestAnalysis,
) -> str:
    """Render the urgent alert sent to the human-manager chat."""
    username = f"@{user.username}" if user.username else "(no username)"
    return (
        "🤬<b>URGENT: Angry Customer Alert!</b>🤬\n"
        f"<b>User</b>: {escape(username)} (ID: <code>{user.id}</code>)\n"
        f"<b>Intent</b>: {escape(analysis.intent)}\n"
        f"<b>Sentiment</b>: {escape(analysis.sentiment)}\n"
        f"<b>Message</b>: {escape(user_text)}"
    )


async def _notify_manager(
    *,
    user: types.User,
    user_text: str,
    analysis: CustomerRequestAnalysis,
) -> None:
    """Send the escalation message to the manager chat.

    Never raises — escalation is best-effort: if the manager chat is
    unreachable, we just log the problem and let the main flow continue.
    """
    if not MANAGER_CHAT_ID:
        logger.warning(
            "Angry customer detected (user_id=%s) but MANAGER_CHAT_ID is unset",
            user.id,
        )
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


@dp.message(F.text)
async def handle_text(message: types.Message) -> None:
    """Hybrid pipeline: intent classifier → structured LLM analyzer → reply (+escalation)."""
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

    if should_route_to_rag(classification) and rag_service is not None:
        try:
            policy_answer = await asyncio.to_thread(rag_service.ask, user_text)
        except Exception:
            logger.exception("RAG policy lookup failed for user_id=%s", message.from_user.id)
            await message.answer(
                "Sorry, I couldn't look up the store policy right now. "
                "Please try again in a moment.",
            )
            return

        try:
            await message.answer(escape(policy_answer), parse_mode="HTML")
            await message.answer(_build_rag_card(intent_hint), parse_mode="HTML")
        except Exception:
            logger.exception("Failed to deliver RAG reply to user_id=%s", message.from_user.id)
        return

    async with ChatActionSender(
        bot=bot,
        chat_id=message.chat.id,
        action=ChatAction.TYPING,
    ):
        analysis = await analyze_request(user_text, intent_hint=intent_hint)

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
        await message.answer(escape(analysis.ai_response_text), parse_mode="HTML")
        if facts_block is not None:
            await message.answer(facts_block, parse_mode="HTML")
        await message.answer(
            _build_extracted_card(
                analysis,
                db_order=db_order,
                classifier_intent=intent_hint,
            ),
            parse_mode="HTML",
        )
    except Exception:
        logger.exception("Failed to deliver reply to user_id=%s", message.from_user.id)
        return

    if analysis.sentiment == "Critical/Angry":
        await _notify_manager(
            user=message.from_user,
            user_text=user_text,
            analysis=analysis,
        )


async def _init_rag() -> None:
    global rag_service
    try:
        rag_service = StorePolicyRAG()
        await asyncio.to_thread(rag_service.build_or_load_index, str(_POLICY_PDF))
        logger.info("Store policy RAG index is ready")
    except Exception:
        logger.exception("Failed to initialize StorePolicyRAG — RAG branch disabled")
        rag_service = None


async def _on_startup() -> None:
    """Boot-time tasks: prepare DB, seed it, and warm up RAG index."""
    await init_db()
    await seed_database()
    await _init_rag()


async def _on_shutdown() -> None:
    """Release resources held by long-lived clients."""
    await intent_client.close()
    await bot.session.close()


async def main() -> None:
    """Entry point: bring up DB and start long polling."""
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
