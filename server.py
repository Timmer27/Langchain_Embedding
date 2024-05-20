from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Flask 애플리케이션 초기화
app = Flask(__name__)

# LangChain 초기화
local_path = '../../models/nous-hermes-llama2-13b.Q4_0.gguf'
prompt = PromptTemplate(
    input_variables=["text"],
    template="Name any five companies which makes `{text}`?",
)

llm = GPT4All(
    model=local_path,
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True,
    verbose=True,
)

chain = prompt | llm | StrOutputParser()

# API 엔드포인트 정의
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "text is required"}), 400

    response = chain.invoke({"text": text})
    print('prit', response)
    return jsonify({"response": response})

# Flask 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
