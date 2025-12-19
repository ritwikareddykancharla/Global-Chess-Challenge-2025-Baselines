MODEL_NAME_OR_PATH="dipamc/chess_qwen_4b_v3_special_tokens"
# MODEL_NAME_OR_PATH="<path/to/your/saved_sft_model>"

vllm serve $MODEL_NAME_OR_PATH \
    --served-model-name aicrowd-chess-model \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --disable-log-stats \
    --host 0.0.0.0 \
    --port 8000