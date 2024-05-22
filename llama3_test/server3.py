## server.py 파일   ### 모델을 다운로드 받아 로컬에 저장해서 사용하는 방식.
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 경고 메시지 무시 설정
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

# Hugging Face 모델 초기화 (강제 다운로드 설정)
tokenizer = AutoTokenizer.from_pretrained("Bllossom/llama-3-Korean-Bllossom-70B", force_download=False)
model = AutoModelForCausalLM.from_pretrained("Bllossom/llama-3-Korean-Bllossom-70B", force_download=False)

# 특수 토큰 임베딩 처리
model.resize_token_embeddings(len(tokenizer))

def generate_with_huggingface(text):
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# API 엔드포인트 정의
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({"error": "text is required"}), 400

    response = generate_with_huggingface(text)
    
    return jsonify({"response": response})

# Flask 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
