from enum import Enum
from typing import Annotated, Literal

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


class InnerStrContent(BaseModel):
    field: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    value: list[Annotated[str, StringConstraints(min_length=1, max_length=128)]]


class InnerAgeDxContent(BaseModel):
    field: Literal["cases.diagnoses.age_at_diagnosis"]
    value: Annotated[int, Field(ge=0, le=32872)]


class InnerYearDxContent(BaseModel):
    field: Literal["cases.diagnoses.year_of_diagnosis"]
    value: Annotated[int, Field(ge=1900, le=2050)]


class InnerCigDayContent(BaseModel):
    field: Literal["cases.exposures.cigarettes_per_day"]
    value: Annotated[int, Field(ge=0, le=999999)]


class InnerPackYrContent(BaseModel):
    field: Literal["cases.exposures.pack_years_smoked"]
    value: Annotated[int, Field(ge=0, le=999999)]


class InnerCigStrtContent(BaseModel):
    field: Literal["cases.exposures.tobacco_smoking_onset_year"]
    value: Annotated[int, Field(ge=1900, le=2050)]


class Inner(BaseModel):
    op: InnerOp
    content: (
        InnerStrContent
        | InnerAgeDxContent
        | InnerYearDxContent
        | InnerCigDayContent
        | InnerPackYrContent
        | InnerCigStrtContent
    )


class Middle(BaseModel):
    op: MiddleOp
    content: list[Inner]


class GDCCohortSchema(BaseModel):
    op: OuterOp
    content: list[Inner | Middle]
