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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import json
import requests
import math
import numpy as np
import os
def dcg(relevance_scores):
    """
    计算折扣累积增益（DCG）
    :param relevance_scores: 一个列表，表示每个文档的相关性分数
    :return: DCG 值
    """
    dcg_value = 0.0
    for i, relevance in enumerate(relevance_scores, start=1):
        dcg_value += (2 ** relevance - 1) / np.log2(i + 1)
    return dcg_value

def ndcg(sorted_docs, golden_answer_list):
    """
    计算归一化折扣累积增益（NDCG）
    :param sorted_docs: 一个列表，表示已经排好序的文档
    :param golden_answer_list: 一个列表，表示所有相关文档（golden answers）
    :return: NDCG 值
    """
    # 将文档映射为相关性分数（在 golden_answer_list 中的文档为 1，否则为 0）
    relevance_scores = [1 if doc in golden_answer_list else 0 for doc in sorted_docs]
    
    # 计算 DCG
    dcg_value = dcg(relevance_scores)
    
    # 计算 IDCG（理想情况下的 DCG，所有相关文档都排在前面）
    ideal_relevance_scores = [1] * len(golden_answer_list) + [0] * (len(sorted_docs) - len(golden_answer_list))
    idcg_value = dcg(ideal_relevance_scores)
    
    # 防止分母为零
    if idcg_value == 0:
        return 0.0
    
    # 计算 NDCG
    ndcg_value = dcg_value / idcg_value
    return ndcg_value

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


class RMManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None,rm_url="http://0.0.0.0:8003/eval") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.rm_url = rm_url
    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score, r_reflect = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        data_eval = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            generated_answer = get_answer_from_predict_str(self.tokenizer.decode(valid_response_ids))
            if generated_answer is None:
                generated_answer = 'Please Judge False'
            data_eval.append(dict(
                query = extra_info['question'],
                generated_answer = generated_answer,
                reference_answer = data_item.non_tensor_batch['reward_model']['ground_truth']
            ))

        data_to_be_eval = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score, r_reflect = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            if score >0.0:
                data_to_be_eval.append(data_eval[i])
        
        if len(data_to_be_eval) > 0:
            request_data_to_be_eval = dict(
                bs=300,
                prompts=data_to_be_eval
            )
            prompts_json = json.dumps(request_data_to_be_eval)
            print("=====================eval model start=====================")
            response = requests.post(self.rm_url, json=prompts_json)
            eval_results = response.json()
            print("=====================eval model end=====================")
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            
            # 从 compute_score 获取初始分数和反射分数
            initial_score, r_reflect = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            final_score = initial_score # 默认最终分数为初始分数
            

            if initial_score > 0.0:
                print("\n==================== DEBUG: START ====================")
                print(f"[DEBUG] initial_score = {initial_score:.4f}")
                print(f"[DEBUG] question = {extra_info.get('question', 'N/A')}")
                print(f"[DEBUG] file_name = {extra_info.get('file_name', 'N/A')}")
                print(f"[DEBUG] reference_page = {extra_info.get('reference_page', 'N/A')}")
                
                # 打印回答信息
                print(f"[DEBUG] Ground Truth Answer: {ground_truth}")
                print(f"[DEBUG] Model Answer (raw): {response_str}")

                generated_answer = get_answer_from_predict_str(response_str)
                print(f"[DEBUG] Model Answer (parsed): {generated_answer}")

                # Safe retrieval
                retrievaled_images = data_item.non_tensor_batch.get('retrievaled_images', [])
                print(f"[DEBUG] raw retrievaled_images = {retrievaled_images}")

                if retrievaled_images:
                    retrievaled_images_basename_list = [
                        os.path.basename(item.rstrip('/')).split(".jpg")[0].replace("_page_", "_") 
                        for item in retrievaled_images
                    ]
                else:
                    retrievaled_images_basename_list = []
                    print("[WARNING] retrievaled_images is empty or missing.")

                print(f"[DEBUG] retrievaled_images_basename_list = {retrievaled_images_basename_list}")

                reference_images_basename_list = []
                if "file_name" in extra_info and "reference_page" in extra_info:
                    reference_images_basename_list = [
                        f'{extra_info["file_name"].split(".pdf")[0]}_{page}' 
                        for page in extra_info["reference_page"]
                    ]
                else:
                    print("[WARNING] file_name or reference_page missing in extra_info")

                print(f"[DEBUG] reference_images_basename_list = {reference_images_basename_list}")

                if retrievaled_images_basename_list:
                    ndcg_value = ndcg(retrievaled_images_basename_list, reference_images_basename_list)
                else:
                    ndcg_value = 0.0

                print(f"[NDCG] value = {ndcg_value:.4f}")
                print("==================== DEBUG: END ====================\n", flush=True)
                print(f"[NDCG] retrievaled={retrievaled_images_basename_list}")
                print(f"[NDCG] reference={reference_images_basename_list}")
                print(f"[NDCG] value={ndcg_value}", flush=True)
                # --- 获取外部模型评估分数 ---
                model_eval_score = eval_results.pop(0)

                # ================= DEBUG PRINTING START =================
                # 在这里打印各个奖励的分数，以便调试
                print("\n" + "="*20 + " DEBUG REWARD COMPONENTS " + "="*20)
                print(f"  - Initial Score (from compute_score) : {initial_score:.4f}")
                print(f"  - Reflect Score (from compute_score) : {r_reflect:.4f}")
                print(f"  - NDCG Score (retrieval quality)     : {ndcg_value:.4f}")
                print(f"  - Model Eval Score (from RM)         : {model_eval_score:.4f}")
                # ================= DEBUG PRINTING END ===================

                # --- 计算最终加权分数 ---
                final_score = (0.6 * model_eval_score + 
                               0.2 * initial_score + 
                               0.1 * ndcg_value + 
                               0.1 * r_reflect)
                
                # 打印加权计算公式
                print(f"  - Weighted Calculation: 0.6*{model_eval_score:.2f} + 0.2*{initial_score:.2f} + 0.1*{ndcg_value:.2f} + 0.1*{r_reflect:.2f} = {final_score:.4f}")
                print("="*65 + "\n", flush=True)


            reward_tensor[i, valid_response_length - 1] = final_score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                # 打印最终使用的分数
                print("[final_score]", final_score) 

        return reward_tensor