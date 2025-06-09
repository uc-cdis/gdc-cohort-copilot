import requests
from fastapi import FastAPI
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
    model = client.models.list().data[0].id

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
    max_len = client.models.list().data[0].max_model_len

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

    cohort_filter = completion.choices[0].text
    filter_instance = GDCCohortSchema.model_validate_json(cohort_filter)
    formatted_filter = filter_instance.model_dump_json(indent=2)

    return Response(cohort=formatted_filter)


app = FastAPI()
app.mount("/api", api)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
