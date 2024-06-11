#!/bin/bash

# 清除屏幕
clear
# 设置运行参数
USE_API=False
API_NAME='glm'
TOP_K=5
PARENT_CHUNK_SIZE=1000
CHUNK_SIZE=200
CHUNK_OVERLAP=50
RERANK_TOP_K=4
IS_FIRST=False
# 执行agent运行脚本
python run_agent.py \
    --api_name $API_NAME \
    --top_k $TOP_K \
    --parent_chunk_size $PARENT_CHUNK_SIZE \
    --chunk_size $CHUNK_SIZE \
    --chunk_overlap $CHUNK_OVERLAP \
    --rerank_top_k $RERANK_TOP_K \
    