#!/bin/bash

set -x

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  # replace it with your local file path
TRAIN_FILE=hiyouga/math12k # replace it with your local file path

python -m verl.trainer.main \
    config=/g22551154lrr/FAST-GRPO-main-69dac76506ba3e2fe2d2ae2387f594d83e4d3124/FAST-GRPO-main-69dac76506ba3e2fe2d2ae2387f594d83e4d3124/examples/config2_kl_group.yaml \
    data.train_files=${TRAIN_FILE} \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=fast-grpo-llm \
    trainer.n_gpus_per_node=4 \
    
