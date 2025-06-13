import requests
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from .schema import GDCCohortSchema

JSON_SCHEMA = GDCCohortSchema.model_json_schema()

api = FastAPI(openapi_prefix="/api")


class Query(BaseModel):
    text: str


class Response(BaseModel):
    cohort: str


@api.post("/cohort")
async def generate_cohort(query: Query) -> Response:
    client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
    model_details = client.models.list().data[0]
    model = model_details.id

    # get length of prompt
    ret = requests.post(
        "http://localhost:8001/tokenize",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "prompt": query.text,
        },
    )

    prompt_len = ret.json()["count"]
    max_len = model_details.max_model_len

    # generate cohort json
    completion = client.completions.create(
        model=model,
        prompt=query.text,
        n=1,
        temperature=0,
        max_tokens=max_len - prompt_len,
        seed=42,
        extra_body={"guided_json": JSON_SCHEMA},
    )

    if completion.choices[0].finish_reason == "length":
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail=(
                f"Model generation did not complete due to insufficient context length. "
                f"Available generation token length = max model token length ({max_len}) - input prompt token length ({prompt_len}) = {max_len - prompt_len}"
            ),
        )
    cohort_filter = completion.choices[0].text
    filter_instance = GDCCohortSchema.model_validate_json(cohort_filter)
    formatted_filter = filter_instance.model_dump_json(indent=4)

    return Response(cohort=formatted_filter)


app = FastAPI()
app.mount("/api", api)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
