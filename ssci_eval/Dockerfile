FROM python:3.12-slim-bookworm

# Lightweight client + offline analysis deps. No GPU needed: this container only
# makes HTTP calls to the vllm service and crunches the cached results.
RUN pip install --no-cache-dir httpx numpy scikit-learn

WORKDIR /work

# Code is bind-mounted at runtime (see docker-compose.eval.yml) so scripts can be
# edited and re-run without rebuilding the image. Stay alive for `exec`.
CMD ["sleep", "infinity"]
