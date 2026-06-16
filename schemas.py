from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


SentimentLiteral = Literal["Positive", "Neutral", "Negative", "Critical/Angry"]


class CustomerRequestAnalysis(BaseModel):

    intent: str = Field(
        description=(
            "Short label of the user's intent in Title Case. "
            "Examples: 'Delivery Complaint', 'Refund Request', 'Order Status Inquiry', "
            "'General Question', 'Product Question', 'Cancellation Request'."
        ),
    )
    sentiment: SentimentLiteral = Field(
        description=(
            "Overall sentiment of the user's message. "
            "Must be EXACTLY one of: Positive, Neutral, Negative, Critical/Angry."
        ),
    )
    order_number: Optional[str] = Field(
        default=None,
        description=(
            "Order number extracted from the user's message (e.g. 'ORD-1001'). "
            "Null if no order number is mentioned. Do not invent or guess it."
        ),
    )
    address: Optional[str] = Field(
        default=None,
        description=(
            "Physical delivery address extracted from the user's message. "
            "Null if no address is mentioned. Do not invent or guess it."
        ),
    )
    phone: Optional[str] = Field(
        default=None,
        description=(
            "Phone number extracted from the user's message, in the form the user wrote it. "
            "Null if no phone number is mentioned. Do not invent or guess it."
        ),
    )
    ai_response_text: str = Field(
        description=(
            "Polite reply to the user, written STRICTLY IN ENGLISH. "
            "If the sentiment is 'Critical/Angry', the tone MUST be deeply apologetic "
            "and the message MUST explicitly state that a human manager has been "
            "notified and will get back to the customer shortly. "
            "Never expose internal tooling, the database, or this analysis schema."
        ),
    )
    wants_overall_summary: bool = Field(
        default=False,
        description=(
            "Set to TRUE only when the user asks about the overall situation across "
            "ALL orders (e.g. 'how many orders are in transit?', 'what's the overall "
            "status?', 'give me a summary', 'how are things going in general?'). "
            "Set to FALSE for questions about a single specific order, complaints, "
            "or anything else. NEVER invent statistics in `ai_response_text` — when "
            "this flag is true, the application will append real numbers from the "
            "database after your reply."
        ),
    )
