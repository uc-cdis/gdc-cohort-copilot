from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class InnerOp(Enum):
    IN = "in"
    EQ = "="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="


class MiddleOp(Enum):
    AND = "and"


class OuterOp(Enum):
    AND = "and"
    EQ = "="


class InnerContent(BaseModel):
    field: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    value: (
        list[Annotated[str, StringConstraints(min_length=1, max_length=128)]]
        | Annotated[float, Field(ge=0, le=32872)]
    )


class Inner(BaseModel):
    op: InnerOp
    content: InnerContent


class Middle(BaseModel):
    op: MiddleOp
    content: list[Inner]


class GDCCohortSchema(BaseModel):
    op: OuterOp
    content: list[Inner | Middle]
