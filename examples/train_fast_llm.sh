#!/bin/bash

set -x

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  # replace it with your local file path
TRAIN_FILE=agentica-org/DeepScaleR-Preview-Dataset # replace it with your local file path

python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${TRAIN_FILE} \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=fast-grpo-llm \
    trainer.n_gpus_per_node=4 \
    
