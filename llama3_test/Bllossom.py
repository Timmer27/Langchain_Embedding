##################################################################################################################
###################################### try 4 - yanolja.py 
##################################################################################################################
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

model_id = "Bllossom/llama-3-Korean-Bllossom-70B"

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

# 시작시간 2024-05-27 17:41:39.948360 device: cuda
# Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
# Loading checkpoint shards:  63%|█████████████████████████████████████████████████████▏                              | 19/30 [15:42<09:19, 50.86s/it]Killed


##################################################################################################################
###################################### try 3 
##################################################################################################################
# Bllossom1.py 파일
# Python code with Pipeline 
# import os
# import transformers
# import torch
# import time
# from datetime import datetime

# # 모델 다운로드 위치 설정
# cache_dir = "/data/models"
# os.environ["TRANSFORMERS_CACHE"] = cache_dir

# start_time = time.time()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("시작시간", datetime.now(), "device:", device)

# # model_id = "MLP-KTLim/llama3-Bllossom"
# model_id = "Bllossom/llama-3-Korean-Bllossom-70B"
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16, "cache_dir": cache_dir},
#     device_map="auto",
# )

# with torch.no_grad():
# #pipeline.model.eval()

# 	PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
# 	You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
# 	instruction = "서울과학기술대학교 MLP연구실에 대해 소개해줘"

# 	messages = [
# 		{"role": "system", "content": f"{PROMPT}"},
# 		{"role": "user", "content": f"{instruction}"}
# 		]

# 	prompt = pipeline.tokenizer.apply_chat_template(
# 			messages, 
# 			tokenize=False, 
# 			add_generation_prompt=True
# 	)
# 	# Debugging prints
#     print("Prompt:", prompt)

# 	terminators = [
# 		pipeline.tokenizer.eos_token_id,
# 		pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# 	]

# 	# Clear CUDA cache to free up memory
# 	torch.cuda.empty_cache()

# 	outputs = pipeline(
# 		prompt,
# #		max_new_tokens=2048,
# 		max_new_tokens=512,
# 		eos_token_id=terminators,
# 		do_sample=True,
# 		temperature=0.6,
# 		top_p=0.9,
# 		repetition_penalty = 1.1
# 	)

# 	print(outputs[0]["generated_text"][len(prompt):])

# # 서울과학기술대학교 MLP연구실은 멀티모달 자연어처리 연구를 하고 있습니다. 구성원은 임경태 교수와 김민준, 김상민, 최창수, 원인호, 유한결, 임현석, 송승우, 육정훈, 신동재 학생이 있습니다.
# # 이것도 결국 다운 받는 버전임. 파일을 삭제하니 500 Internal Server Error 에러남. 파일사이즈도 늘어나고..

# end_time = time.time()

# execution_time = end_time - start_time

# hours, remainder = divmod(execution_time, 3600)
# minutes, seconds = divmod(remainder, 60)

# # 결과 출력
# print("종료시간", datetime.now())
# print(f"수행 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.6f}초")

##################################################################################################################
###################################### try 2 
##################################################################################################################

# import transformers
# import torch
# import os
# import time
# from datetime import datetime

# # 모델 다운로드 위치 설정
# cache_dir = "/data/models"
# os.environ["TRANSFORMERS_CACHE"] = cache_dir

# start_time = time.time()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("시작시간", datetime.now(), "device:", device)

# model_id = "Bllossom/llama-3-Korean-Bllossom-70B"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16, "cache_dir": cache_dir},
#     device_map="auto",
# )

# pipeline.model.eval()
# # with torch.no_grad():

# PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
# You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
# instruction = "서울과학기술대학교 MLP연구실에 대해 소개해줘"

# messages = [
# 	{"role": "system", "content": f"{PROMPT}"},
# 	{"role": "user", "content": f"{instruction}"}
# 	]

# prompt = pipeline.tokenizer.apply_chat_template(
# 		messages, 
# 		tokenize=False, 
# 		add_generation_prompt=True
# )

# terminators = [
# 	pipeline.tokenizer.eos_token_id,
# 	pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = pipeline(
# 	prompt,
# 	max_new_tokens=2048,
# 	eos_token_id=terminators,
# 	do_sample=True,
# 	temperature=0.6,
# 	top_p=0.9,
# )

# print(outputs[0]["generated_text"][len(prompt):])
 
# end_time = time.time()
# # 서울과학기술대학교 MLP연구실은 멀티모달 자연어처리 연구를 하고 있습니다. 구성원은 임경태 교수와 김민준, 김상민, 최창수, 원인호, 유한결, 임현석, 송승우, 육정훈, 신동재 학생이 있습니다.
# # 이것도 결국 다운 받는 버전임. 파일을 삭제하니 500 Internal Server Error 에러남. 파일사이즈도 늘어나고..

# execution_time = end_time - start_time

# hours, remainder = divmod(execution_time, 3600)
# minutes, seconds = divmod(remainder, 60)

# # 결과 출력
# print("종료시간", datetime.now())
# print(f"수행 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.6f}초") 

###################################### original --> magicdb에서는 수행됨.
# Python code with AutoModel

# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # 모델 다운로드 위치 설정
# cache_dir = "/data/models"
# os.environ["TRANSFORMERS_CACHE"] = cache_dir

# start_time = time.time()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("시작시간", datetime.now(), "device:", device)

# model_id = 'MLP-KTLim/llama3-Bllossom'
# model_id = "Bllossom/llama-3-Korean-Bllossom-70B"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16, "cache_dir": cache_dir},
#     device_map="auto",
# )

# model.eval()
# # with torch.no_grad():
PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
# instruction = "윤석열에 대해 소개해줘"

# messages = [
# 	{"role": "system", "content": f"{PROMPT}"},
# 	{"role": "user", "content": f"{instruction}"}
# 	]

# input_ids = tokenizer.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	return_tensors="pt"
# ).to(model.device)

# terminators = [
# 	tokenizer.eos_token_id,
# 	tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = model.generate(
# 	input_ids,
# 	max_new_tokens=2048,
# 	eos_token_id=terminators,
# 	do_sample=True,
# 	temperature=0.6,
# 	top_p=0.9,
# 	repetition_penalty = 1.1
# )

# print(tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))
# # 서울과학기술대학교 MLP연구실은 멀티모달 자연어처리 연구를 하고 있습니다. 구성원은 임경태 교수와 김민준, 김상민, 최창수, 원인호, 유한결, 임현석, 송승우, 육정훈, 신동재 학생이 있습니다.