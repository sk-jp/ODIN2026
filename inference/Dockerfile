FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel AS wheel_builder

ENV MAX_JOBS=4 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install --yes --no-install-recommends \
        build-essential git ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip wheel --no-cache-dir --no-deps --no-build-isolation --wheel-dir /wheels \
        causal-conv1d==1.6.2.post1 && \
    python -m pip wheel --no-cache-dir --no-deps --no-build-isolation --wheel-dir /wheels \
        mamba-ssm==2.3.2.post1


FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/opt/app/benchmark_networks:/opt/app

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

COPY --from=wheel_builder /wheels /wheels
COPY requirements.txt /opt/app/

RUN python -m pip install --no-cache-dir --no-color --requirement requirements.txt && \
    python -m pip install --no-cache-dir --no-color /wheels/causal_conv1d*.whl /wheels/mamba_ssm*.whl && \
    python -c "from causal_conv1d import causal_conv1d_fn; from fla.modules import FusedRMSNormGated; from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule" && \
    rm -rf /wheels

RUN apt-get update && apt-get install --yes --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY --chown=user:user inference.py nnUNetPlans.json dataset.json /opt/app/
COPY --chown=user:user toothfairy2_feature_extraction /opt/app/toothfairy2_feature_extraction
COPY --chown=user:user benchmark_networks /opt/app/benchmark_networks

ENV PYTORCH_ALLOC_CONF=expandable_segments:True

USER user

ENTRYPOINT ["python", "inference.py"]
