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

This repository contains the official implementation of **FAST-GRPO** (Fast-Slow Thinking Group Relative Policy Optimization).

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
| FAST-1.5B | DeepSeek-R1-Distill-Qwen-1.5B | [ModelScope](https://modelscope.cn/models/xiaowenyi/FAST-1.5B) |
| FAST-3B | Qwen-2.5-VL-3B | [ModelScope](https://modelscope.cn/models/xiaowenyi/FAST-3B) |
| FAST-7B | Qwen-2.5-VL-7B | Coming Soon |
| FAST-4B | Qwen-3-VL-4B | Coming Soon |

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
- This repository provides a reimplementation using [EasyR1]([https://github.com/EasyR1/EasyR1](https://github.com/hiyouga/EasyR1)) framework
- Thanks to the [VeRL](https://github.com/volcengine/verl) and [EasyR1]([https://github.com/EasyR1/EasyR1](https://github.com/hiyouga/EasyR1)) team for the base training framework.
