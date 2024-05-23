# Hugging Face에서 제공하는 pipeline을 사용하는 방법 (로컬파일 활용x)
from flask import Flask, request, jsonify
import transformers
import torch

app = Flask(__name__)

# 모델 아이디 설정
model_id = "Bllossom/llama-3-Korean-Bllossom-70B"

# 파이프라인 초기화
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
# 텍스트 생성 함수
def generate_with_pipeline(text):
    outputs = pipeline(
        text,
        max_length=204800,
        truncation=True,  # truncation 옵션을 명시적으로 설정
    )
    return outputs[0]['generated_text']

# API 엔드포인트 정의
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({"error": "text is required"}), 400

    print("text :", text)
    response = generate_with_pipeline(text)[:1024]
    # Setting `pad_token_id` to `eos_token_id`:144783 for open-end generation 에러 아님. 
    # https://velog.io/@castle_mi/GPT -> 굉장히 오래 걸림
    # 이것도 결국 다운 받는 버전임. 파일을 삭제하니 500 Internal Server Error 에러남. 파일사이즈도 늘어나고..
    
    return jsonify({"response": response})

# Flask 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
