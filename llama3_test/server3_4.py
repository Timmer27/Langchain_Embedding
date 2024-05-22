# LangChain + llama-3-Korean-Bllossom
from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import pipeline
import torch

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 모델 로드
model_id = "Bllossom/llama-3-Korean-Bllossom-70B"
pipeline_model = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

prompt = PromptTemplate(
    input_variables=["text"],
    template="Name any five companies which makes `{text}`?",
)

llm = GPT4All(
    model=pipeline_model,
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True,
    verbose=True,
)
# chain = prompt | llm | StrOutputParser()
chain = prompt | pipeline_model | StrOutputParser()

# API 엔드포인트 정의
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({"error": "text is required"}), 400

    print("text :", text)
    # 토크나이징
    encoded_input = pipeline_model.tokenizer(text, return_tensors="pt")

    # 추론 수행
    with torch.no_grad():
        output = pipeline_model(**encoded_input)

    # 응답 생성
    response = pipeline_model.tokenizer.decode(output.logits[0], clean_up=True)
    
    return jsonify({"response": response})

# Flask 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

