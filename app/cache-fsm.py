import sys

assert len(sys.argv) == 2  # input model path on cmd line

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from schema import GDCCohortSchema  # isort: skip

JSON_SCHEMA = GDCCohortSchema.model_json_schema()

# the path of the model must precisely match the path used during inference
llm = LLM(sys.argv[1])

# the sampling params do not need to match, only the JSON schema
sampling_params = SamplingParams(
    n=1,
    temperature=0,
    max_tokens=128,
    seed=42,
    guided_decoding=GuidedDecodingParams(json=JSON_SCHEMA),
)

# run inference so FSM is computed and cached
out = llm.generate(["TCGA cases"], sampling_params)
