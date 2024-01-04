#!/bin/bash

# 这里可以放入代码运行命令
echo "program start..."
cd /app
python agent_predict.py \
    --question_file_path '/tcdata/question_v2.json' \
    --answer_file_path 'submit_result.jsonl' \
    --random_seed 47