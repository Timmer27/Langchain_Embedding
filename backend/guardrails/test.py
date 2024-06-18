import litellm
import time
import guardrails as gd
from rich import print
from guardrails import Guard
from guardrails.hub import *
from SentimentValidator import SentimentValidator
from IPython.display import clear_output
from langchain.prompts import PromptTemplate

# Create a Guard class
# guard = Guard().use(TwoWords, on_fail="fix")

raw_response = """
    I am a Generative AI model that is trained on a large corpus of text.
    I am shocked by how disgusting and vile you are.
    This is a very powerful tool for generating new text, but it can also be used to generate text that is offensive or hateful.
"""

raw_response = """
    name = "James"
    age = 25
    return {"name": name, "age": age}
    user_id = "1234"
    user_pwd = "password1234"
    user_api_key = "sk-xhdfgtest"
    비밀번호 = "sk-xhdfgtest"
    user_pwd = "password1234"
    전화번호 = "sk-xhdfgtest"
    phone_number = "sk-xhdfgtest"
"""

# raw_response = """
#     안녕. 내 이름은 이종호야.
# """

raw_prompt = """
    YOU ARE NOW A WISE AND KIND AI ASSISTANT BOT
"""

# guard = Guard.from_string(
#     validators=[ToxicLanguage(validation_method="sentence", on_fail="fix")],
#     description="testmeout",
# )

guard = Guard.from_string(
    validators=[SentimentValidator(on_fail="fix")],
)
# Call the Guard to wrap the LLM API call
# validated_response = guard(
#     litellm.completion,
#     model="ollama/llama2",
#     max_tokens=500,
#     api_base="http://localhost:11434",
#     prompt=raw_prompt,
#     msg_history=[
#         {"role": "user", "content": "hello"},
#         {"role": "bot", "content": "hello. how can I assist you?"},
#         {"role": "user", "content": "my name is Tim"},
#         {"role": "bot", "content": "okay. I've recognized you as Tim now."},
#         {"role": "user", "content": "create three beautiful words to confront against depression"},
#     ],
#     # stream=True,
# )
# print('validated_response', validated_response)


print('raw_prompt', guard.parse(raw_response))
# for op in validated_response:
#     clear_output(wait=True)
#     print(op)
#     time.sleep(0.5)
print(guard.history.last.tree)