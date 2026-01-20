## Deploy 起來後，預設會在 http://localhost:8000/v1/chat/completions 提供服務
CUDA_VISIBLE_DEVICES=7 \
swift deploy \
  --use_hf true \
  --infer_backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapters ./output/llama31_8b_scam_real_sft_v4/v0-20260117-075831/checkpoint-108 \
  --served_model_name scam-8b-sft \
  --max_new_tokens 1


#for test normal 
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hi, what time is our meeting tomorrow? Reply: 2pm."}'
#for test scam
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have won a prize! Please provide your bank account and password, we will transfer the money to you."}'
