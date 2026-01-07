from typing import Annotated

from sqlalchemy import BigInteger
from sqlalchemy.orm import mapped_column


IntPK = Annotated[
    int,
    mapped_column(
        BigInteger,
        primary_key=True,
    ),
]
