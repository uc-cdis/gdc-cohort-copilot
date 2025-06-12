import sys

assert len(sys.argv) == 2  # input model path on cmd line

from vllm.model_executor.guided_decoding.outlines_decoding import JSONLogitsProcessor
from vllm.transformers_utils.tokenizer import get_tokenizer

from schema import GDCCohortSchema  # isort: skip

JSON_SCHEMA = GDCCohortSchema.model_json_schema()

# the path of the model/tok must precisely match the path used during inference
tok = get_tokenizer(sys.argv[1])

# compile FSM through vllm/outlines utils
JSONLogitsProcessor(
    schema=JSON_SCHEMA,
    tokenizer=tok,
    whitespace_pattern=None,
)
