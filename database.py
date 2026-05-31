"""SQLite-хранилище заказов с авто-инициализацией и сидингом тестовыми данными.

Используется LangChain-инструментами агента (см. ``agent.py``).
Через ``aiosqlite`` всё работает асинхронно и дружит с aiogram-loop'ом.

Данные внутри (имена, товары, статусы) — на английском, потому что пользователи
общаются с ботом на английском, и LLM-агент тоже отвечает по-английски.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Final

import aiosqlite


logger = logging.getLogger(__name__)

DB_PATH: Final[Path] = Path(__file__).parent / "orders.db"

_SEED_ORDERS: Final[list[dict[str, str]]] = [
    {
        "order_number": "ORD-1001",
        "status": "In transit",
        "items": "Sony WH-1000XM5 wireless headphones (x1)",
        "customer_name": "John Peterson",
    },
    {
        "order_number": "ORD-1002",
        "status": "Delivered",
        "items": "DeLonghi Magnifica S coffee machine (x1)",
        "customer_name": "Anna Smith",
    },
    {
        "order_number": "ORD-1003",
        "status": "Processing at warehouse",
        "items": "'Clean Code' book (x2), leather bookmark (x2)",
        "customer_name": "Michael Brown",
    },
    {
        "order_number": "ORD-1004",
        "status": "In transit",
        "items": "Logitech G502 gaming mouse (x1), SteelSeries QcK mousepad (x1)",
        "customer_name": "Catherine Wilson",
    },
    {
        "order_number": "ORD-1005",
        "status": "Cancelled",
        "items": "Xiaomi 14 smartphone (x1)",
        "customer_name": "Daniel Roberts",
    },
    {
        "order_number": "ORD-1006",
        "status": "Delivered",
        "items": "Bosch electric kettle (x1)",
        "customer_name": "Olivia Newman",
    },
    {
        "order_number": "ORD-1007",
        "status": "Awaiting payment",
        "items": "LG UltraGear 27GP850 monitor (x1)",
        "customer_name": "Steven Morris",
    },
    {
        "order_number": "ORD-1008",
        "status": "Processing at warehouse",
        "items": "XD Design Bobby backpack (x1), Anker 20000 power bank (x1)",
        "customer_name": "Victoria Lambert",
    },
]


async def init_db() -> None:
    """Создать файл БД и таблицу ``orders``, если их ещё нет."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_number   TEXT    NOT NULL UNIQUE,
                    status         TEXT    NOT NULL,
                    items          TEXT    NOT NULL,
                    customer_name  TEXT    NOT NULL
                )
                """,
            )
            await db.commit()
        logger.info("База данных инициализирована: %s", DB_PATH)
    except aiosqlite.Error:
        logger.exception("Не удалось инициализировать БД")
        raise


async def seed_database() -> None:
    """Залить тестовые заказы, если таблица пуста."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute("SELECT COUNT(*) FROM orders") as cursor:
                row = await cursor.fetchone()
                count: int = row[0] if row else 0

            if count > 0:
                logger.info("Сидинг пропущен: в таблице уже %d заказов", count)
                return

            await db.executemany(
                """
                INSERT INTO orders (order_number, status, items, customer_name)
                VALUES (:order_number, :status, :items, :customer_name)
                """,
                _SEED_ORDERS,
            )
            await db.commit()
        logger.info("Сидинг выполнен: добавлено %d заказов", len(_SEED_ORDERS))
    except aiosqlite.Error:
        logger.exception("Ошибка при сидинге БД")
        raise


async def fetch_order_by_number(order_number: str) -> dict[str, Any] | None:
    """Достать один заказ по номеру; ``None`` если не найден."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT id, order_number, status, items, customer_name
                FROM orders
                WHERE order_number = ?
                """,
                (order_number.strip(),),
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None
    except aiosqlite.Error:
        logger.exception("Ошибка при чтении заказа %s", order_number)
        raise


async def fetch_orders_summary() -> dict[str, Any]:
    """Вернуть агрегированную статистику по всем заказам."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT status FROM orders",
            ) as cursor:
                rows = await cursor.fetchall()
    except aiosqlite.Error:
        logger.exception("Ошибка при сборе статистики по заказам")
        raise

    statuses: list[str] = [row["status"] for row in rows]
    by_status: dict[str, int] = dict(Counter(statuses))

    return {
        "total_orders": len(statuses),
        "by_status": by_status,
    }
