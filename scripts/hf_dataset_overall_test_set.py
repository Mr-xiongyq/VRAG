import json
from datasets import Dataset
import os
import datasets
import argparse
from tqdm import tqdm


# FEWSHOT_EXAMPLES = '''
# example1:
# <think>
# The image shows a timeline of Nestlé acquisitions from 2000 to 2011. The question asks for the brand acquired in 2000. I will crop the area near the left edge of the timeline, close to the year 2000.
# </think>
# <bbox>[66, 236, 125, 350]</bbox>
# <answer>Purina</answer>

# <reflection>
# My initial selection was too far to the right, leading to the incorrect identification of Purina, which was acquired in 2001. I should shift my focus to the very beginning of the timeline to correctly locate the 2000 acquisition.
# </reflection>

# <think>
# Based on the reflection, I will crop the far left area of the timeline. The brand logo directly aligned with the year 2000 is PowerBar.
# </think>
# <bbox>[10, 240, 80, 360]</bbox>
# <answer>PowerBar</answer>

# example2:
# <think>
# The image shows Nestlé's acquisition timeline from 2000 to 2011. I want to find the brand acquired in 2000. I will crop the area between 2000 and 2001.
# </think>
# <bbox>[66, 236, 125, 350]</bbox>
# <answer>Purina</answer>

# <reflection>
# Although my cropped region included both 2000 and 2001, I mistakenly identified the 2001 brand as the 2000 acquisition. On closer inspection, the smaller bar on the left is actually labeled 2000 and corresponds to PowerBar. I misunderstood the visual layout.
# </reflection>

# <think>
# After reflecting, I realize that the 2000 acquisition is represented by the short bar at the very left, with the PowerBar logo. Purina is placed above the taller 2001 bar.
# </think>
# <answer>PowerBar</answer>
# '''

# USER_PROMPT = '''Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want.

# After each answer, you must always include a <reflection> ... </reflection> block to review and verify the correctness of your reasoning and image selection, even if your answer seems correct.

# Then, you must revise your plan by providing a new <think> step and a new <answer> based on your reflection, if necessary.

# If you find no further external knowledge is needed, and your reasoning is complete, you can directly provide the final answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.

# Question: {question}'''


# def get_final_prompt(question):
#     return f"""{USER_PROMPT.format(question=question)}

# Here are some examples to guide your reasoning:

# {FEWSHOT_EXAMPLES}"""


# # all_examples = [example for example in all_examples if example['query'] not in sft_questions]
# def convert_dataset(file_list,file_source_list,output_name):
#     all_examples = []
#     for file_name, source_type in zip(file_list, file_source_list):
#         with open(file_name, "r") as f:
#             file_data = json.load(f)
#             data_list = file_data["examples"]
#             for item in data_list:
#                 item['source'] = source_type
#             all_examples.extend(data_list)

#     for example in all_examples:
#                 # 初始化默认值
#         example['content_type'] = 'Nan'
#         example['reason_type'] = 'Nan'
#         if example['source'] == 'vidoseek':
#             example['reason_type'] = example['meta_info']['query_type']
#             example['content_type'] = example['meta_info']['source_type']
#         elif example['source'] == 'slidevqa_test':
#             query_type = example['meta_info'].get('query_type', '')
#             if 'Multi-Hop' in query_type:
#                 example['reason_type'] = 'MultiHop'
#             elif 'Single-Hop' in query_type:
#                 example['reason_type'] = 'SingleHop'
#             else:
#                 example['reason_type'] = 'Nan'
                        
#             # 对于slidevqa_test，根据你的JSON示例，看起来没有span类型信息
#             # 如果有其他规则来确定content_type，请在这里添加
#             if 'Non-Span' in query_type:
#                 example['content_type'] = 'NonSpan'
#             elif 'Single-Span' in query_type:
#                 example['content_type'] = 'SingleSpan'
#             elif 'Multi-Span' in query_type:
#                 example['content_type'] = 'MultiSpan'
#             else:
#                 example['content_type'] = 'Nan'  # 或者设置为其他默认值
    
#         elif example['source'] == 'mmlongdoc':
#             example['content_type'] = '####'.join(example['meta_info']['source_type'])
#             example['reason_type'] = example['meta_info']['doc_type']
#         else:
#             example['content_type'] = 'Nan'
#             example['reason_type'] = 'Nan'

#     dataset = Dataset.from_dict({
#         "id": [str(example["uid"]) for example in all_examples],
#         "problem": [example["query"] for example in all_examples],
#         "prompt": [get_final_prompt(example["query"]) for example in all_examples],
#         "answer": [example["reference_answer"] for example in all_examples],
#         "file_name": [example["meta_info"]["file_name"] for example in all_examples],
#         "reference_page": [
#             example["meta_info"].get("reference_page") 
#             if "reference_page" in example["meta_info"] 
#             else "####".join(example["meta_info"].get("image_local_name", []))
#             for example in all_examples
#         ],
#         "data_source_type": [example["source"] for example in all_examples],
#         "query_content_type": [example["content_type"] for example in all_examples],
#         "query_reason_type": [example["reason_type"] for example in all_examples]
#     })

#     def make_map_fn_test(split):
#         def process_fn(example, idx):
#             prompt = example.pop('prompt')
#             answer = example.pop('answer')
#             problem = example.pop('problem')
#             data_source = example.pop('data_source_type')
#             reference_page = example.pop('reference_page')
#             file_name = example.pop('file_name')
#             # images = example.pop('images')
#             query_content_type = example.pop('query_content_type')
#             query_reason_type = example.pop('query_reason_type')

#             data = {
#                 "data_source": data_source,
#                 "prompt": [{
#                     "role": "user",
#                     "content": prompt,
#                 }],
#                 # "images": images,
#                 "ability": "math",
#                 "reward_model": {
#                     "style": "rule",
#                     "ground_truth": answer
#                 },
#                 "extra_info": {
#                     'split': split,
#                     'index': idx,
#                     'answer': answer,
#                     "question": problem,
#                     "content_type": query_content_type,
#                     "reason_type": query_reason_type,
#                     "file_name": file_name,
#                     "reference_page": reference_page
#                 }
#             }
#             return data
#         return process_fn

#     test_dataset = dataset.map(function=make_map_fn_test('test'), with_indices=True, num_proc=8)

#     test_dataset.to_parquet(f'./data/rag/{output_name}.parquet')

# if __name__ == '__main__':
#     convert_dataset(
#         ['/data3/xiongyuqi/SlideVQA/rag_dataset_raw_train.json'],
#         ['slidevqa_test'],
#         'slidevqa_train_crop_3'
#     )
# # if __name__ == '__main__':
# #     convert_dataset(
# #         ['/data3/xiongyuqi/SlideVQA/rag_dataset_raw_val.json'],
# #         ['slidevqa_val'],
# #         'overall_test_crop_3'
# #     )

# USER_PROMPT = '''Answer the given question step by step using the following structure:

# 1. You **must always start** with reasoning inside <think>...</think>.
# 2. After reasoning, **you must always perform a <search> query </search>**, unless the question is purely factual and you are absolutely certain.
# 3. After search, you may choose to use image coordinates via <bbox>[x1, y1, x2, y2]</bbox> to crop or refine your results.
# 4. Then, provide your answer inside <answer>...</answer>.
# 5. After every answer, you **must include a <reflection>...</reflection>** block to review and verify your reasoning and retrieved content.
# 6. Based on your reflection, start a new <think> and <answer> cycle if needed.

# ⚠️ **Never skip the <search> step unless you are 100% confident** that no external information is needed.

# Example (final form if no more steps are needed):  
# <answer> Beijing </answer>

# Question: {question}
# '''
# Question: {question}
# '''
USER_PROMPT = '''You are a reasoning agent that must follow a strict multi-step structure to answer a question.  
Your response **must strictly follow this order** in each reasoning cycle:

1. <think>...</think>  
   - Start with internal reasoning. Think about what is needed to answer the question.

2. <search>...</search>  
   - Based on your reasoning, perform a search to retrieve relevant information.

3. <bbox>[x1, y1, x2, y2]</bbox>  
   - If needed, specify a bounding box to crop part of an image or document for better focus.  
     If no cropping is needed, still include an empty tag like <bbox></bbox>.

4. <answer>...</answer>  
   - Give your answer based on the retrieved and/or cropped information.

5. <reflection>...</reflection>  
   - Reflect on whether your reasoning, search results, and answer are correct and complete.

---

If your reflection identifies missing information or an incorrect answer, you **must** start another full cycle of:

<think>...</think>  
<search>...</search>  
<bbox>...</bbox>  
<answer>...</answer>  
<reflection>...</reflection>  

Repeat this loop until your final <reflection> confirms that the answer is complete and correct.

---

### Rules:
- You must **never skip** any of the 5 steps in each cycle.
- You must include `<bbox></bbox>` even if no cropping is needed.
- You must not generate `<answer>` before completing `<search>` and `<bbox>`.
- Every `<answer>` must be followed by a `<reflection>`.

---

### Example (Final Output if no further steps are needed):

<think> I need to find the capital of China. </think>  
<search> capital of China </search>  
<bbox>[66,236,80,350]</bbox>  
<answer> Beijing </answer>  
<reflection> The answer is complete and no further search is needed. </reflection>

---

Now, follow the structure to answer the question below.

Question: {question}
'''

import json
from datasets import Dataset
import os
import datasets
import argparse
from tqdm import tqdm


def convert_dataset(USER_PROMPT, file_list, file_source_list, output_name):
    all_examples = []
    for file_name, source_type in zip(file_list, file_source_list):
        with open(file_name, "r") as f:
            file_data = json.load(f)
            data_list = file_data["examples"]
            for item in data_list:
                item['source'] = source_type
            all_examples.extend(data_list)

    # 处理每个示例，添加content_type和reason_type
    for example in all_examples:
        # 初始化默认值
        example['content_type'] = 'Nan'
        example['reason_type'] = 'Nan'
        
        if example['source'] == 'vidoseek':
            example['reason_type'] = example['meta_info'].get('query_type', 'Nan')
            example['content_type'] = example['meta_info'].get('source_type', 'Nan')
        elif example['source'] == 'slidevqa_test':
            query_type = example['meta_info'].get('query_type', '')
            if 'Multi-Hop' in query_type:
                example['reason_type'] = 'MultiHop'
            elif 'Single-Hop' in query_type:
                example['reason_type'] = 'SingleHop'
            else:
                example['reason_type'] = 'Nan'
            
            # 对于slidevqa_test，根据你的JSON示例，看起来没有span类型信息
            # 如果有其他规则来确定content_type，请在这里添加
            if 'Non-Span' in query_type:
                example['content_type'] = 'NonSpan'
            elif 'Single-Span' in query_type:
                example['content_type'] = 'SingleSpan'
            elif 'Multi-Span' in query_type:
                example['content_type'] = 'MultiSpan'
            else:
                example['content_type'] = 'Nan'  # 或者设置为其他默认值
                
        elif example['source'] == 'mmlongdoc':
            source_types = example['meta_info'].get('source_type', [])
            if isinstance(source_types, list):
                example['content_type'] = '####'.join(source_types)
            else:
                example['content_type'] = str(source_types)
            example['reason_type'] = example['meta_info'].get('doc_type', 'Nan')
        
        # 如果没有匹配的source类型，保持默认的'Nan'值

    dataset = Dataset.from_dict({
        "id": [str(example["uid"]) for example in all_examples],
        "problem": [example["query"] for example in all_examples],
        "prompt": [USER_PROMPT.replace('{question}', example["query"]) for example in all_examples],
        "answer": [example["reference_answer"] for example in all_examples],
        "file_name": [example["meta_info"]["file_name"] for example in all_examples],
        "reference_page": [
            example["meta_info"].get("reference_page") 
            if "reference_page" in example["meta_info"] 
            else "####".join(example["meta_info"].get("image_local_name", []))
            for example in all_examples
        ],
        "data_source_type": [example["source"] for example in all_examples],
        "query_content_type": [example["content_type"] for example in all_examples],
        "query_reason_type": [example["reason_type"] for example in all_examples]
    })

    def make_map_fn_test(split):
        def process_fn(example, idx):
            prompt = example.pop('prompt')
            answer = example.pop('answer')
            problem = example.pop('problem')
            data_source = example.pop('data_source_type')
            reference_page = example.pop('reference_page')
            file_name = example.pop('file_name')
            query_content_type = example.pop('query_content_type')
            query_reason_type = example.pop('query_reason_type')

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": problem,
                    "content_type": query_content_type,
                    "reason_type": query_reason_type,
                    "file_name": file_name,
                    "reference_page": reference_page
                }
            }
            return data
        return process_fn

    test_dataset = dataset.map(function=make_map_fn_test('test'), with_indices=True, num_proc=8)
    test_dataset.to_parquet(f'./data/rag/{output_name}.parquet')


# if __name__ == '__main__':
#     convert_dataset(
#         USER_PROMPT,
#         ['/data3/xiongyuqi/SlideVQA/rag_dataset_raw_train.json'],
#         ['slidevqa_train'],
#         'slidevqa_train_crop'
#     )
# if __name__ == '__main__':
#     convert_dataset(
#         USER_PROMPT,
#         ['/data3/xiongyuqi/SlideVQA/rag_dataset_raw_test.json', '/data3/xiongyuqi/ViDoSeek/vidoseek.json'],
#         ['slidevqa_test', 'vidoseek'],
#         'overall_test_baseline'
#     )

# if __name__ == '__main__':
#     convert_dataset(
#         USER_PROMPT,
#         ['/data3/xiongyuqi/SlideVQA/rag_dataset_raw_train.json'],
#         ['slidevqa_test'],
#         'slidevqa_train_crop_3'
#     )
if __name__ == '__main__':
    convert_dataset(
        USER_PROMPT,
        ['/data3/xiongyuqi/SlideVQA/rag_dataset_raw_val.json'],
        ['slidevqa_val'],
        'overall_test_crop_3'
    )