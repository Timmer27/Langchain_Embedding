import os
import configparser
from flask import Flask, Response, request, jsonify
import threading
import queue
from flask_cors import CORS
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import GPT4All
# from langchain_community.llms import Ollama
from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from langchain.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from src.utils import load_persisted_chroma_db
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")


# from langchain.chains import RunnableSequence, RunnableLambda, RunnablePassthrough
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnablePassthrough

app = Flask(__name__)
CORS(app)

load_dotenv()
def load_api_key():
    config = configparser.ConfigParser()
    config.read('config.ini')
    api_key = config['API']['OPENAI_API_KEY']
    return api_key

def set_api_key_env_var(api_key):
    os.environ["OPENAI_API_KEY"] = api_key

class UserQuery(BaseModel):
    """user question input model"""
    question: str
    
class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)

def llm_OpenAI(g, prompt):
    try:
        # data = request.get_json()
        # query = UserQuery(**data)
        # chat = ChatOpenAI(
        #     verbose=True,
        #     streaming=True,
        #     callbacks=[ChainStreamHandler(g)],
        #     temperature=0.7,
        # )
        # chat([HumanMessage(content=prompt)])
        model = ChatOpenAI(
            verbose=True,
            streaming=True,
            callbacks=[ChainStreamHandler(g)],
            temperature=0.7,
        )
        # Model 설정
        # model = OpenAI(
        #     model_name="gpt-3.5-turbo-instruct",
        #     temperature=0.2,
        #     max_tokens=512,
        #     streaming=True,
        #     callbacks=[ChainStreamHandler(g)]
        # )
        # Vector Store 설정
        print("Vector Store 설정")
        
        db = Chroma(persist_directory="./vector_store", embedding_function=OpenAIEmbeddings())
        retriever = db.as_retriever(search_type="similarity")
        
        # print("retriever ------------> ")
        # print(retriever)
        # print("format_docs")
        # print(format_docs)
        
        # llm_chain = PromptTemplate(input_variables=["text"], template=prompt) | model | StrOutputParser()
        # llm_chain.invoke({"text": ""})
        # 되는 버전 ------------------------
        llm_chain = (
        # llm_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}    # TypeError: argument 'text': 'dict' object cannot be converted to 'PyString'
        # # llm_chain = ( {"text": retriever | format_docs, "question": RunnablePassthrough()}  # TypeError: argument 'text': 'dict' object cannot be converted to 'PyString'
            # PromptTemplate(template="Your question: {question}")                          
            PromptTemplate(input_variables=["context", "question"], template=prompt)    # OK
        # #     # PromptTemplate(input_variables=["context", "question"], template="{context}\n\n{question}")
        #     # PromptTemplate(input_variables=["text", "question"], template="{prompt}\n\n{question}") # KeyError: "Input to PromptTemplate is missing variables {'prompt'}.  Expected: ['prompt', 'question'] Received: ['text', 'question']" 
        #     # PromptTemplate(input_variables=["text", "question"], template=prompt)     # OK 
        #     # | PromptTemplate(input_variables=["text"], template=prompt)               # OK 
        #     # | prompt      # OK but 안녕 -> 죄송합니다. 해당 질문에 대해서는 답변을 할 수 없습니다. 다른 질문을 해주세요.
            | model 
            | StrOutputParser()
        )
        # llm_chain.invoke({"text": ""})                    # OK 
        # llm_chain.invoke({"text": "", "question": ""})      # OK 
        llm_chain.invoke({"context": "", "question": ""})      # OK 
        # # Chain 실행
        # # response = llm_chain.invoke({"question": user_prompt})
        # answer = llm_chain.invoke(query.question).strip()
        # for token in answer:
        #     g.send(token)
        # print("llm_chain.invoke ------------> ")

        
    finally:
        g.close()

def llm_gpt4(g, prompt):
    try:
        local_path = './models/nous-hermes-llama2-13b.Q4_0.gguf'
        model = GPT4All(
            model=local_path,
            callbacks=[ChainStreamHandler(g)],
            streaming=True,
            verbose=True,
        )
        llm_chain = PromptTemplate(input_variables=["text"], template=prompt) | model
        llm_chain.invoke({"text": ""})
    finally:
        g.close()

def llm_Ollama(g, prompt):
    try:
        local_path = os.path.abspath('./models/nous-hermes-llama2-13b.Q4_0.gguf')  # Use absolute path
        model = Ollama(
            model=local_path,
            callbacks=[ChainStreamHandler(g)],
            verbose=True,
        )
        llm_chain = PromptTemplate(input_variables=["text"], template=prompt) | model
        llm_chain.invoke({"text": ""})
    finally:
        g.close()        

def llm_openAI_with_chroma(g, prompt):
    # Chroma DB를 로드합니다.
    db = load_persisted_chroma_db()
    # 로드된 DB를 이용하여 Retriever를 초기화합니다.
    retriever = db.as_retriever(search_type="similarity")
    # Retriever를 호출 시, custom callback을 포함합니다.
    docs = retriever.invoke(prompt, {"callbacks": ChainStreamHandler(g)})
    print('docs', docs)
    g.close()
    return docs

def chain(prompt):
    g = ThreadedGenerator()
    # threading.Thread(target=llm_OpenAI, args=(g, prompt)).start()
    threading.Thread(target=llm_openAI_with_chroma, args=(g, prompt)).start()
    # threading.Thread(target=llm_Ollama, args=(g, prompt)).start()
    return g

def load_chunk_persist_pdf(dataPath) -> Chroma:
    data = []
    for file in os.listdir(dataPath):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(dataPath, file)
            loader = PyPDFLoader(pdf_path)
            data.extend(loader.load())
    # 데이터 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, 
        chunk_overlap = 0
    )
    documents = text_splitter.split_documents(data)

    db = Chroma.from_documents(persist_directory="vector_store", documents=documents, embedding=OpenAIEmbeddings())
    return db

@app.route('/dbtest', methods=['GET'])
def dbtest():
    # db = load_chunk_persist_pdf('./data')
    db = load_persisted_chroma_db()
    retriever = db.as_retriever()
    docs = retriever.invoke("경제전망 어케 되냐")
    if docs:
        return docs[0].page_content
    else:
        return "No relevant documents found."

@app.route('/chain', methods=['POST'])
def _chain():
    return Response(chain(request.json['prompt']), mimetype='text/plain')     # OK
    # query = request.json["question"]                                          # KeyError: 'question'
    # answer = chain.invoke(query).strip()
    # return jsonify({"answer": answer})

if __name__ == '__main__':
    api_key = load_api_key()
    set_api_key_env_var(api_key)
    app.run(host='0.0.0.0', port=5001, threaded=True, debug=True)
