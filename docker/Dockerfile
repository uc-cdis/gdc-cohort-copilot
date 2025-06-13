FROM vllm/vllm-openai:v0.6.4.post1

# need to install fastapi cli utilities to launch app server
RUN pip install fastapi[standard]

# patch outlines so caching with vllm works
# see the following github issues/PRs:
    # https://github.com/dottxt-ai/outlines/pull/1129
    # https://github.com/dottxt-ai/outlines/issues/1145
    # https://github.com/dottxt-ai/outlines/issues/1130
RUN sed -i 's/return args_to_key(base, args, kwargs, typed, ignore)/return str(args_to_key(base, args, kwargs, typed, ignore))/' /usr/local/lib/python3.12/dist-packages/outlines/caching.py

# copy model and util files
ENV PROJ_PATH="/gdc-cohort-pilot"
COPY utils/cache-fsm.py $PROJ_PATH/app/cache-fsm.py
COPY utils/schema.py $PROJ_PATH/app/schema.py
COPY --from=model . $PROJ_PATH/model

# set workdir so relative imports in scipts resolve
WORKDIR $PROJ_PATH/app
ENV IN_CONTAINER_MODEL_PATH="../model"

# compute outlines cache
ENV OUTLINES_CACHE_DIR=".outlines-cache"
RUN python3 cache-fsm.py $IN_CONTAINER_MODEL_PATH

# copy app - separating this allows faster rebuild for developing frontend
COPY docker/app $PROJ_PATH/app

# set entrypoint to wrapper script that launches app and vllm servers in parallel
ENTRYPOINT ["bash", "run.sh"]
