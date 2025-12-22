# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import heapq
from calendar import c
import re
from typing import Any
import numpy as np
from scipy.stats import rankdata

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def normalize_image_complexity(complexities: list[float]) -> np.ndarray:
    """Normalize image complexities to [0, 1] based on ranking."""
    if not complexities or all(c == complexities[0] for c in complexities):
        return np.ones(len(complexities)) * 0.5
    
    ranks = rankdata(complexities, method='average')
    normalized = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else np.array([0.5])
    return normalized


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    K_ROLLOUTS = 8  
    num_responses = len(reward_inputs)
    
    if num_responses == 0:
        return []

    processed_data = []    
    problem_pass_rates = [] 
    problem_complexities = []
    sum_len = 0
    current_problem_correct_count = 0
    current_problem_rollout_count = 0
    current_problem_complexity = None
    
    for idx, reward_input in enumerate(reward_inputs):
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        length = len(response)
        sum_len += length
        acc = accuracy_reward(response, reward_input["ground_truth"])
        fmt = format_reward(response)
        
        # Extract image complexity if available
        image_complexity = reward_input.get("image_complexity", 0.5)
        
        # Store complexity for current problem
        if current_problem_rollout_count == 0:
            current_problem_complexity = image_complexity
        
        processed_data.append({
            "acc": acc, 
            "fmt": fmt, 
            "len": length,
            "image_complexity": image_complexity
        })
        
        if acc == 1.0:
            current_problem_correct_count += 1
        
        current_problem_rollout_count += 1

        if current_problem_rollout_count == K_ROLLOUTS:
            pass_rate = current_problem_correct_count / K_ROLLOUTS
            problem_pass_rates.append(pass_rate)
            problem_complexities.append(current_problem_complexity)
            current_problem_correct_count = 0
            current_problem_rollout_count = 0
            current_problem_complexity = None
            
    if current_problem_rollout_count > 0:
        pass_rate = current_problem_correct_count / K_ROLLOUTS
        problem_pass_rates.append(pass_rate)
        if current_problem_complexity is not None:
            problem_complexities.append(current_problem_complexity)

    avg_len = sum_len / num_responses if num_responses > 0 else 0.0

    # Normalize image complexities to [0, 1] based on ranking
    if problem_complexities:
        normalized_complexities = normalize_image_complexity(problem_complexities)
    else:
        normalized_complexities = np.array([0.5] * len(problem_pass_rates))
    
    # Compute combined difficulty scores
    problem_difficulties = []
    for i, pass_rate in enumerate(problem_pass_rates):
        # Combine pass rate with normalized image complexity
        # Higher complexity and lower pass rate = higher difficulty
        difficulty = (1 - pass_rate) * normalized_complexities[i]
        problem_difficulties.append(difficulty)
    
    if not problem_difficulties:
        difficulty_threshold = 0.5
    else:
        # Problems with difficulty above 80th percentile are considered difficult
        difficulty_threshold = np.percentile(problem_difficulties, 80)

    scores = []
    punish_count = len(reward_inputs) * 0.8
    
    for idx, data in enumerate(processed_data):
        problem_idx = idx // K_ROLLOUTS
        
        current_problem_difficulty = problem_difficulties[problem_idx] if problem_idx < len(problem_difficulties) else 0.5
        acc_score = data["acc"]
        format_score = data["fmt"]
        
        len_score = 0.0
        
        # Problem is difficult if its difficulty score is above threshold
        is_difficult_problem = current_problem_difficulty > difficulty_threshold
        
        # Reward efficiency for easy problems (low difficulty)
        if not is_difficult_problem and acc_score == 1.0:
            if punish_count > 0:
                punish_count -= 1
                len_score = 1 - data["len"] / avg_len
        
        # Reward effort for difficult problems
        if acc_score == 0 and is_difficult_problem:
            len_score = min(data["len"] / avg_len - 1, 1)
        
        overall_score = acc_score + 0.5 * format_score + 0.5 * len_score
        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "accuracy": acc_score,
                "difficulty": current_problem_difficulty,
                "image_complexity": data["image_complexity"]
            }
        )

    return scores