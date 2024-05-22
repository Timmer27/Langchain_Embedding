# huggingface Python code with Pipeline 소스 적극 활용
from flask import Flask, request, jsonify

import transformers
import torch

# Flask 애플리케이션 초기화
app = Flask(__name__)

# LangChain 초기화
local_path = '../models/nous-hermes-llama2-13b.Q4_0.gguf'
model_id = "Bllossom/llama-3-Korean-Bllossom-70B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)



# API 엔드포인트 정의
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "text is required"}), 400

    pipeline.model.eval()

    PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
    You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
    instruction = "서울과학기술대학교 MLP연구실에 대해 소개해줘"

    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{instruction}"}
        ]
    print("messages ", messages)
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    print("prompt ", messages)

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    print("outputs = pipeline -------> ")
    # pipeline.generation_config.pad_token_id = pipeline.generation_config.eos_token_id
    # AttributeError: 'TextGenerationPipeline' object has no attribute 'generation_config'

    outputs = pipeline(
        prompt,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    # Setting `pad_token_id` to `eos_token_id`:144783 for open-end generation

    response = outputs[0]["generated_text"][len(prompt):]
    print('prit', response)
    return jsonify({"response": response})

# Flask 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
