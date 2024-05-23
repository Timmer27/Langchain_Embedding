# Bllossom1.py 파일
# Python code with Pipeline 
import os
import transformers
import torch
import time
from datetime import datetime

# 모델 다운로드 위치 설정
cache_dir = "/nas2/models"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

start_time = time.time()
print("시작시간", datetime.now())

# model_id = "MLP-KTLim/llama3-Bllossom"
model_id = "Bllossom/llama-3-Korean-Bllossom-70B"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16, "cache_dir": cache_dir},
    device_map="auto",
)

pipeline.model.eval()

PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
instruction = "서울과학기술대학교 MLP연구실에 대해 소개해줘"

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty = 1.1
)

print(outputs[0]["generated_text"][len(prompt):])

# 서울과학기술대학교 MLP연구실은 멀티모달 자연어처리 연구를 하고 있습니다. 구성원은 임경태 교수와 김민준, 김상민, 최창수, 원인호, 유한결, 임현석, 송승우, 육정훈, 신동재 학생이 있습니다.
# 이것도 결국 다운 받는 버전임. 파일을 삭제하니 500 Internal Server Error 에러남. 파일사이즈도 늘어나고..

end_time = time.time()

execution_time = end_time - start_time

hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)

# 결과 출력
print("종료시간", datetime.now())
print(f"수행 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.6f}초")