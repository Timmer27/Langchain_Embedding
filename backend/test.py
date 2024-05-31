from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from src.utils import initialize_chroma_db
import configparser
import os

config = configparser.ConfigParser()
config.read('config.ini')
os.environ['OPENAI_API_KEY'] = config['API']['OPENAI_API_KEY']

# 해당 모델은 한국어 안됨
# model = "Meta-Llama-3-8B-Instruct:latest"
# model = "Llama-3-Open-Ko-8B-Instruct:latest"
model = "EEVE-Korean-10.8B:latest"
template = """
    {question}
"""

db = initialize_chroma_db('./data')

llm = ChatOllama(
    model=model,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

# # 프롬프트 설정
# prompt = ChatPromptTemplate.from_template("한국어 할줄 아니?")

# # LangChain 표현식 언어 체인 구문을 사용합니다.
# chain = prompt | llm | StrOutputParser()
# _res = chain.invoke({"question": "한국어 할줄 아니?"})

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=template,
            input_variables=["question"],
        ),
        "document_variable_name": "question"
    },
)

_res = chain.invoke({"question": "엔비디아의 대표 상품에 대해 알려줘"})
# _res = chain.invoke({"question": "한국어 할줄 아니?"})
