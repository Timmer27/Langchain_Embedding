# Use a pipeline as a high-level helper
import os
import transformers
import torch
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 다운로드 위치 설정
cache_dir = "/data/models"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

start_time = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("시작시간", datetime.now(), "device:", device)

model_id = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

# Initialize the tokenizer and model separately
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(device)

# Initialize the pipeline with the model and tokenizer
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
)

prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {prompt}\nAssistant:\n"
text = '한국의 수도는 어디인가요? 아래 선택지 중 골라주세요.\n\n(A) 경성\n(B) 부산\n(C) 평양\n(D) 서울\n(E) 전주'
model_inputs = tokenizer(prompt_template.format(prompt=text), return_tensors='pt').to(device)

# Generate text using the model
outputs = model.generate(**model_inputs, max_new_tokens=256)
output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(output_text)

end_time = time.time()
execution_time = end_time - start_time
hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)

# 결과 출력
print("종료시간", datetime.now())
print(f"수행 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.6f}초")



# Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text-generation", model="yanolja/EEVE-Korean-Instruct-10.8B-v1.0")

# from transformers import AutoTokenizer
# from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0")
# tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0")

# prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {prompt}\nAssistant:\n"
# text = '한국의 수도는 어디인가요? 아래 선택지 중 골라주세요.\n\n(A) 경성\n(B) 부산\n(C) 평양\n(D) 서울\n(E) 전주'
# model_inputs = tokenizer(prompt_template.format(prompt=text), return_tensors='pt')

# outputs = model.generate(**model_inputs, max_new_tokens=256)
# output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
# print(output_text)