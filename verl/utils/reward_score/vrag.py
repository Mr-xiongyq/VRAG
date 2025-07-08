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

import re
def get_answer_from_predict_str(text):
    end_tag = '</answer>'
    start_tag = '<answer>'
    
    end_pos = text.rfind(end_tag)
    if end_pos == -1:
        return None  # 如果没有找到</answer>，返回None
    
    start_pos = text.rfind(start_tag, 0, end_pos)
    if start_pos == -1:
        return None  # 如果没有找到<answer>，返回None
    
    start_pos += len(start_tag)  # 跳过<answer>标签
    return text[start_pos:end_pos]

def calculate_anls(gold_labels, prediction, threshold=0.7):
    import numpy as np
    from Levenshtein import distance as levenshtein_distance
    max_scores = []
    for gold_label in gold_labels:
        ld = levenshtein_distance(prediction, gold_label)
        max_len = max(len(prediction), len(gold_label))
        if max_len == 0:
            nld = 0.0
        else:
            nld = ld / max_len
        if nld < threshold:
            score = 1 - nld
        else:
            score = 0.0
        max_scores.append(score)
    return max(max_scores)

def compute_reflection_reward(predict_str: str, ground_truth: str) -> float:
    """
    根据是否反思、是否修正成功，计算反思奖励。
    """
    from Levenshtein import distance as levenshtein_distance

    answers = re.findall(r'<answer>(.*?)</answer>', predict_str, re.DOTALL)
    reflection_used = '<reflection>' in predict_str

    if not answers:
        return 0.0

    first_answer = answers[0].strip()
    ld_first = levenshtein_distance(first_answer, ground_truth)
    first_correct = (ld_first / max(len(first_answer), len(ground_truth))) < 0.3

    final_answer = answers[-1].strip()
    ld_final = levenshtein_distance(final_answer, ground_truth)
    final_correct = (ld_final / max(len(final_answer), len(ground_truth))) < 0.3

    if first_correct:
        return 1.0
    elif not first_correct and final_correct and reflection_used:
        return 0.5
    elif not first_correct and not final_correct and reflection_used:
        return 0.1
    elif not first_correct and not reflection_used:
        return 0.0
    elif first_correct and not final_correct and reflection_used:
        return -0.25
    return 0.0

def remove_text_between_tags(text):
    pattern = r'<\|im_start\|>user.*?<\|im_end\|>'
    result = re.sub(pattern, '', text)
    return result

# def compute_format_reward_only(predict_str: str, ground_truth: str, extra_info) -> float:
#     predict_str = remove_text_between_tags(predict_str)

#     # 检查必要的结构标签
#     answer_pattern = re.compile(r'<answer>.*?</answer>', re.DOTALL)
#     search_pattern = re.compile(r'<search>.*?</search>', re.DOTALL)
#     reflection_pattern = re.compile(r'<reflection>.*?</reflection>', re.DOTALL)

#     has_answer = re.search(answer_pattern, predict_str) is not None
#     has_search = re.search(search_pattern, predict_str) is not None
#     has_reflection = re.search(reflection_pattern, predict_str) is not None

#     if has_answer and has_search and has_reflection:
#         return 1.0
#     else:
#         return 0.0
# def compute_format_reward_only(predict_str: str, ground_truth: str, extra_info) -> float:
#     predict_str = remove_text_between_tags(predict_str)
#     answer_pattern = re.compile(r'<answer>.*</answer>', re.DOTALL)
#     search_pattern = re.compile(r'<search>.*</search>', re.DOTALL)
#     answer_match = re.search(answer_pattern, predict_str)
#     search_match = re.search(search_pattern, predict_str)
#     return 1.0 if answer_match and search_match else 0.0

# def compute_format_reward_only(predict_str: str, ground_truth: str, extra_info) -> float:
#     predict_str = remove_text_between_tags(predict_str)

#     tag_patterns = {
#         'search': re.compile(r'<search>.*?</search>', re.DOTALL),
#         'answer': re.compile(r'<answer>.*?</answer>', re.DOTALL),
#         'reflection': re.compile(r'<reflection>.*?</reflection>', re.DOTALL),
#     }

#     score = 0.0
#     for tag, pattern in tag_patterns.items():
#         if re.search(pattern, predict_str):
#             score += 1.0

#     return score / len(tag_patterns)  # 结果 ∈ [0.0, 1.0]
# def compute_score(predict_str: str, ground_truth: str, extra_info) -> tuple:
#     """
#     主函数：返回 (回答准确得分, 反思得分)
#     - 如果没有 <search> 则直接返回 0 分（模型未使用检索）
#     - 回答准确度结合 format_reward（结构越规范越加分）
#     """
#     predict_str = remove_text_between_tags(predict_str)

#     # --- 结构标签检测 ---
#     has_search = bool(re.search(r'<search>.*?</search>', predict_str, re.DOTALL))
#     has_answer = bool(re.search(r'<answer>.*?</answer>', predict_str, re.DOTALL))
#     has_reflection = bool(re.search(r'<reflection>.*?</reflection>', predict_str, re.DOTALL))

#     if not has_search:
#         # ❗️没搜索就不给 reward，或者设得非常低
#         return 0.0, 0.0

#     # --- 格式奖励 ---
#     format_reward_value = (int(has_search) + int(has_answer) + int(has_reflection)) / 3.0

#     # --- 提取答案 ---
#     answer = get_answer_from_predict_str(predict_str)
#     if answer is None:
#         return 0.0, 0.0


#     # --- 反思得分 ---
#     r_reflect = compute_reflection_reward(predict_str, ground_truth)

#     return format_reward_value, r_reflect

def compute_score(predict_str: str, ground_truth: str, extra_info) -> tuple:
    """
    奖励函数：仅根据结构完整性 + 反思得分进行奖励，不考虑语义相似度
    返回: (结构格式得分, 反思得分)
    """

    import re

    predict_str = remove_text_between_tags(predict_str)

    # --- 提取各结构标签的出现位置 ---
    think_spans = re.findall(r'<think>.*?</think>', predict_str, re.DOTALL)
    search_spans = re.findall(r'<search>.*?</search>', predict_str, re.DOTALL)
    answer_spans = re.findall(r'<answer>.*?</answer>', predict_str, re.DOTALL)
    reflection_spans = re.findall(r'<reflection>.*?</reflection>', predict_str, re.DOTALL)

    # --- 计算最大轮数 ---
    max_rounds = max(len(think_spans), len(search_spans), len(answer_spans), len(reflection_spans))

    if max_rounds == 0:
        return 0.0, 0.0  # 没有任何结构，直接 0 分

    # --- 结构部分分计算（每轮 0~1） ---
    structure_scores = []
    for i in range(max_rounds):
        count = 0
        if i < len(think_spans): count += 1
        if i < len(search_spans): count += 1
        if i < len(answer_spans): count += 1
        if i < len(reflection_spans): count += 1
        structure_scores.append(count / 4.0)

    # --- 最终结构格式奖励为多轮平均 ---
    format_reward_value = sum(structure_scores) / max_rounds  # 结果 ∈ [0, 1]

    # --- 反思修正奖励 ---
    r_reflect = compute_reflection_reward(predict_str, ground_truth)

    return format_reward_value, r_reflect

# def compute_score(predict_str: str, ground_truth: str, extra_info) -> tuple:
#     """
#     主函数：返回 (回答准确得分, 反思得分)，格式不对则都为 0
#     """
#     predict_str = remove_text_between_tags(predict_str)
#     format_reward_value = compute_format_reward_only(predict_str, ground_truth, extra_info)

#     # if format_reward_value != 1.0:
#     #     return 0.0, 0.0

#     answer = get_answer_from_predict_str(predict_str)
#     if answer is None:
#         return 0.0, 0.0

#     anls_score = calculate_anls([ground_truth], answer, 0.5)
#     r_reflect = compute_reflection_reward(predict_str, ground_truth)
#     return anls_score, r_reflect
def compute_score(predict_str: str, ground_truth: str, extra_info) -> tuple:
    """
    主函数：返回 (回答准确得分, 反思得分)，其中回答得分 anls 包含格式奖励成分
    """
    predict_str = remove_text_between_tags(predict_str)
    format_reward_value = compute_format_reward_only(predict_str, ground_truth, extra_info)

    answer = get_answer_from_predict_str(predict_str)
    if answer is None:
        return 0.0, 0.0

    # raw_anls = calculate_anls([ground_truth], answer, 0.5)
    # final_anls = raw_anls * format_reward_value

    r_reflect = compute_reflection_reward(predict_str, ground_truth)
    return format_reward_value, r_reflect