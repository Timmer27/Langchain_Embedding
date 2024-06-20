import configparser
import statistics
config = configparser.ConfigParser()
config.read('config.ini')

import pandas as pd
import os
os.environ["OPENAI_API_KEY"] = config['API']['OPENAI_API_KEY']

from deepeval.metrics import *
from deepeval.test_case import *

from deepeval.benchmarks import MMLU, HellaSwag
from deepeval.benchmarks.tasks import MMLUTask, HellaSwagTask

from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from src.utils import initialize_chroma_db

# 해당 모델은 한국어 안됨
# model = "Meta-Llama-3-8B-Instruct:latest"
# model = "Llama-3-Open-Ko-8B-Instruct:latest"
model = "EEVE-Korean-10.8B:latest"
template = """
    {question}
"""

mmlu_benchmark = MMLU().load_benchmark_dataset(MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE)
data = [{'input': mmlu.input, 'expected_output': mmlu.expected_output} for mmlu in mmlu_benchmark]

db = initialize_chroma_db('./data')

llm = ChatOllama(
    model=model,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

# # 프롬프트 설정
# prompt = ChatPromptTemplate.from_template("한국어 할줄 아니?")

template = """
    You are a wise and precise chatbot.
    With the below given questions, you will have to solve the question and answer the question instructed with several options to select
    {question}
    Provide only the letter corresponding to your answer (e.g., "A", "B", "C", etc.).
"""

llm_chain = (                     
    PromptTemplate(input_variables=["question"], template=template)
    | model 
    | StrOutputParser()
)

metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' indicates the given 'expected output'",
        "It does not matter whether the 'actual output' has full context or not.",
        "If indicating to correct 'expected output' answer must be OK"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
) 


test_result = []
for evaluate in data:
    # actual_output = llm_chain.invoke({"question": "It depends, some might consider the cat, while others might argue the dog."})
    actual_output = llm_chain.invoke({"question": evaluate['input']})
    # actual_output = "A"

    test_case = LLMTestCase(
        input=evaluate['input'],
        actual_output=actual_output,
        expected_output=evaluate['expected_output']
    )

    metric.measure(test_case)
    result = {'metric_score' : metric.score, 'metric_reason': metric.reason}
    test_result.append(result)
    break

test_result
# test_result

# chain = RetrievalQAWithSourcesChain.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=db.as_retriever(),
#     return_source_documents=True,
#     chain_type_kwargs={
#         "prompt": PromptTemplate(
#             template=template,
#             input_variables=["question"],
#         ),
#         "document_variable_name": "question"
#     },
# )

# metric = HallucinationMetric(threshold=0.5)

# df = pd.read_csv('./evaluation/foreign_affairs.csv')
# test_result = []
# for idx, row in df.iterrows():
#     actual_output = chain.invoke({"question": row['questions']})
    
#     test_case = LLMTestCase(
#         input=row['questions'],
#         actual_output=actual_output,
#         expected_output=row['answer']
#     )

#     metric.measure(test_case)

#     result = {'metric_score' : metric.score, 'metric_reason': metric.reason}
#     test_result.append(result)
#     break

pd.DataFrame(test_result).to_csv('result.csv', encoding='utf-8-sig')

avg = statistics.mean([eval['metric_score'] for eval in test_result])
print(f"모델: ChatOpenAI\nGEval 평가 평균: {avg}")