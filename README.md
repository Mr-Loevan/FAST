<!-- # FAST-GRPO -->

<p align="center">
  <h2 align="center">[NeurIPS 2025 Spotlight] Fast-Slow Thinking GRPO for Large Visual-Language Model Reasoning
</h2>
  <p align="center">
    </br>
        <a href="https://arxiv.org/pdf/2504.18458">
        <img src='https://img.shields.io/badge/Paper-Arxiv-orange' alt='Paper PDF'></a>
        <a href="https://modelscope.cn/collections/FAST-dcb152452a8847">
        <img src='https://img.shields.io/badge/Model-ModelScope-blue' alt='Model'></a>
        <a href="https://github.com/Mr-Loevan/FAST-GRPO">
        <img src='https://img.shields.io/badge/Code-GitHub-green' alt='Code'></a>
  </p>
</p>

## Overview

This repository contains the official implementation of **FAST-GRPO** (Fast-Slow Thinking Group Relative Policy Optimization), achieving high performance in applying fast-slow thinking to both visual and textual reasoning.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Training](#training)
- [Model Zoo](#model-zoo)
- [Citation](#citation)


## Installation

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/Mr-Loevan/FAST-GRPO.git
cd FAST-GRPO

# Create conda environment
conda create -n fast_grpo python=3.11
conda activate fast_grpo

# Install dependencies (Refer to EasyR1 installation)
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
# Run training with default configuration
bash examples/train_fast_llm.sh
```

## Core Components

FAST-GRPO introduces three key innovations that work together to achieve fast-slow reasoning:

<details>
<summary><b>1. Thinking Reward Function</b></summary>

The Thinking Reward Function (`examples/reward_function/thinking_reward.py`) implements an adaptive difficulty-aware reward mechanism:

- **Adaptive Difficulty**: `difficulty = (1 - pass_rate) * normalized_complexity`
- **Differentiated Rewards**: 
  - Easy problems (< 80th percentile) and correct answer: Rewards concise solutions
  - Hard problems (> 80th percentile) and incorrect answer: Rewards exploration efforts

</details>

<details>
<summary><b>2. Dynamic KL Penalty</b></summary>

Implements group-based adaptive KL divergence control for stable training:

```yaml
# Configuration in config.yaml
algorithm:
  kl_penalty: low_var_kl
  kl_coef: 1.0e-2
  kl_type: "group_accuracy_based"
  kl_min_coef: 0.001  # β_min
  kl_max_coef: 0.01   # β_max
```

- **Group-based Adaptation**: Adjusts KL coefficient based on group performance
</details>

<details>
<summary><b>3. Slow2Fast Sampling</b></summary>

Progressive curriculum learning that gradually increases training difficulty:

```yaml
# Configuration in config.yaml
algorithm:
  online_filtering: true
  filter_key: accuracy
  dynamic_filter_schedule:
    - epoch_ratio: 0.5   
      filter_low: 0.3    
      filter_high: 0.99  
    - epoch_ratio: 1.0   
      filter_low: 0.01  
      filter_high: 0.7   
```

- **Phase 1 (0-50%)**: Learn from medium-to-high difficulty samples for slow thinking
- **Phase 2 (50-100%)**: Include easy samples for fast-thinking
</details>

## Training


### Run Training Example

```bash
# Use provided script (recommended)
bash examples/train_fast_llm.sh
```


## Model Zoo

| Model | Base Model | Download |
|-------|------------|----------|
| FAST-1.5B | DeepSeek-R1-Distill-Qwen-1.5B | [ModelScope](https://modelscope.cn/models/ruiruiL/FAST_DS_1_5b) |
| FAST-3B | Qwen-2.5-VL-3B | [ModelScope](https://modelscope.cn/models/xiaowenyi/FAST-3B) |
| FAST-7B | Qwen-2.5-VL-7B | [ModelScope](https://modelscope.cn/models/xiaowenyi/FAST-7B) |
| FAST-8B-Preview | Qwen-3-VL-8B | [ModelScope](https://modelscope.cn/models/xiaowenyi/FAST-8B-Preview) |
| FAST-8B | Qwen-3-VL-8B | Coming Soon |

> **Note:** FAST-8B-Preview is trained on only 10k data points from [ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K). Training of FAST-8B is ongoing.

## Evaluation Results


### Performance on Textual Reasoning Benchmarks

| Method | GSM8K (Acc) | GSM8K (Length) | MATH 500 (Acc) | MATH 500 (Length) | AIME 2024 (Acc) | AIME 2024 (Length) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FAST-1.5B** | **86.8** | **851** | **85.8** | **2645** | **34.17** | **8003** |

> **Note:** Length denotes the number of generated tokens.

### Performance on Visual Reasoning Benchmarks

| Benchmark | Qwen3-VL-8B (Acc) | Qwen3-VL-8B (Length) | GRPO (Acc) | GRPO (Length) | FAST-8B-Preview (Accuracy) | FAST-8B-Preview (Length) |
|---|---:|---:|---:|---:|---:|---:|
| MathVerse | 42.9 | 1768.2 | 81.2 | 1750.3 | 81.6 | 622.5 |
| MathVista | 68.3 | 804.2 | 72.0 | 894.0 | 73.5 | 371.9 |
| CLEVR | 89.0 | 304.2 | 88.0 | 592.1 | 91.0 | 204.1 |
| Dynamath | 62.7 | 1134.7 | 76.5 | 1235.1 | 77.6 | 495.5 |
| Geo3k | 58.6 | 1680.2 | 70.2 | 1973.8 | 70.7 | 639.0 |
| LogicVista | 49.4 | 2078.0 | 62.9 | 1890.4 | 60.9 | 713.9 |
| MathVision | 21.7 | 3007.5 | 45.5 | 3007.6 | 52.7 | 1245.7 |
| MMMUpro | 27.0 | 1722.3 | 51.9 | 1813.3 | 51.9 | 737.4 |
| MMK12 | 51.7 | 2096.4 | 75.5 | 2045.6 | 79.1 | 864.4 |
| WeMath | 64.0 | 1536.6 | 83.4 | 1476.1 | 82.1 | 468.4 |
| A-OKvqa | 63.6 | 394.3 | 86.3 | 384.6 | 87.8 | 158.7 |

> **Note:** Length denotes the number of generated tokens. FAST-8B-Preview is trained on only 10k data points from [ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K). The evaluation is adapted from [PAPO-Eval](https://github.com/xhguo7/PAPO-Eval).

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{xiao2025fastslow,
  title={Fast-Slow Thinking {GRPO} for Large Vision-Language Model Reasoning},
  author={Wenyi Xiao and Leilei Gan},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=MI1uT5rReV}
}
```

## License

This project is licensed under the Apache 2.0 License.

## Acknowledgments

- The results reported in our paper were originally implemented with [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- This repository provides a reimplementation using [EasyR1](https://github.com/hiyouga/EasyR1) framework
- Thanks to the [VeRL](https://github.com/volcengine/verl) and [EasyR1](https://github.com/hiyouga/EasyR1) team for the base training framework.
