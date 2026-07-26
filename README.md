# Scam Call LLM (Fraud Detection)

This project uses LLMs to detect scam call content in real-time. The system includes a VLLM inference engine and a FastAPI backend service.

## Quick Start

### 1. Environment Setup (.env)
Create a `.env` file in the project root directory and add your Hugging Face and GitHub Tokens:

```env
HF_TOKEN=your_HuggingFace_Token
GITHUB_TOKEN=your_GitHub_Token
```
> **Note**: `GITHUB_TOKEN` is used to download Adapter weights from a Private Repository.

### 2. GPU Configuration
Ensure the GPU configuration in `docker-compose.yml` matches your hardware environment:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0  # Specify the GPU ID to use
```

> **IMPORTANT**: VLLM requires at least **90% of available VRAM** to load the model and weights on startup. If VRAM is insufficient, the service will fail to start.

### 3. Launch Services
Build and start the services using Docker Compose:

```bash
docker compose up --build
```

Once started, the API will be available at `http://localhost:9000`.

## Testing

The project includes a testing script `run_script.sh` to verify if the API correctly distinguishes between scam and normal conversations.

```bash
# Test Normal Conversation (Expected Output: False)
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hi, what time is our meeting tomorrow? Reply: 2pm."}'

# Test Scam Conversation (Expected Output: True)
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have won a prize! Please provide your bank account and password, we will transfer the money to you."}'
```

## Evaluation

This repository includes two complementary evaluation paths:

- `evaluation.py` computes the length- and class-weighted DWA metric for saved
  full-conversation inference results across model families and checkpoints.
- [`ssci_eval/`](ssci_eval/README.md) evaluates streaming-prefix decisions,
  replays the Streaming Scam Confidence Index, calibrates alert thresholds,
  analyzes caller-identity sensitivity, and reports time to detection.

The SSCI integration is evaluation-only: it reuses the existing vLLM service and
shared `real_data/` splits, but does not change the production `/predict` API.
The committed calibration currently recommends a `0.50` threshold for
`non_contact` callers; the legacy `0.65` constant is retained for reproducibility
until a separate production decision is made.

Start the evaluation runner with the existing vLLM service:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.eval.yml \
  up -d vllm ssci-eval
```

See [`ssci_eval/README.md`](ssci_eval/README.md) for the full pipeline, cached
artifacts, metrics, limitations, and reproduction commands.
