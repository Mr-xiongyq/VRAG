import requests
import json
n=128
queries = ["what's the highest man in londan"] * n
reference_answers = ["mike"] * n
generated_answers = ["john mike hh"] * n
prompts = [
    dict(
        query=query,
        reference_answer=reference_answer,
        generated_answer=generated_answer
    ) for query, reference_answer, generated_answer in zip(queries, reference_answers, generated_answers)
]
request_json=dict(
    prompts=prompts,
    bs=1000
)
prompts_json = json.dumps(request_json)
response = requests.post("http://0.0.0.0:8003/eval", json=prompts_json)

print(response)
print(response.json())  # 打印 JSON 响应