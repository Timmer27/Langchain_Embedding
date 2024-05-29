import os
import random
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
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from langchain.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from werkzeug.utils import secure_filename
from langchain.chains import RetrievalQA
from src.utils import load_persisted_chroma_db, train_pdf_chroma_db, initialize_chroma_db
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from deepeval.models.base_model import DeepEvalBaseLLM
import warnings
warnings.filterwarnings("ignore")

# # Determine which environment we're in
# flask_env = os.getenv('FLASK_ENV', 'development')

# # Load the appropriate .env file
# if flask_env == 'production':
#     load_dotenv()
# else:
#     load_dotenv('.env.development')
load_dotenv()
app = Flask(__name__)
# mongo_url = os.getenv("URL")
mongo_url = "localhost"
print(">?>>>>>>>>>>>>>", mongo_url)
client = MongoClient(f'mongodb://{mongo_url}:27017/')
CORS(app)


UPLOAD_FOLDER = 'data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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
        _res = llm_chain.invoke({"context": "", "question": ""})      # OK 
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
    # 학습된 pdf 파일명, 경로에 따라 학습할지 말지를 결정
    # 학습이 안된 파일이 있다면, 잠시 시간이 걸리면서 vector로 변환
    # front에서 loading state로 ui 띄워주면 좋을듯?? 
    db = initialize_chroma_db('./data')
    # 로드된 DB를 이용하여 Retriever 초기화
    model = ChatOpenAI(
            verbose=True,
            streaming=True,
            callbacks=[ChainStreamHandler(g)],
            temperature=0.7,
        )
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True,
    )
    _res = chain.invoke(prompt)
    g.close()

def chain(prompt, modal):
    g = ThreadedGenerator()
    if modal == '1':
        threading.Thread(target=llm_OpenAI, args=(g, prompt)).start()
    elif modal == '2':
        threading.Thread(target=llm_gpt4, args=(g, prompt)).start()
    elif modal == '3':
        threading.Thread(target=llm_Ollama, args=(g, prompt)).start()
    else:
        threading.Thread(target=llm_openAI_with_chroma, args=(g, prompt)).start()

    return g

@app.route('/chain/<modal>', methods=['POST'])
def _chain(modal):
    return Response(chain(request.json['prompt'], modal), mimetype='text/plain')     # OK

@app.route('/model/<modalId>', methods=['GET'])
def edit_modals(modalId):
    db = client['vector_files']
    file_collection = db["file_collection"]
    object_id = ObjectId(modalId)
    document = file_collection.find_one({"_id": object_id})
    # for document in cursor:
    #     modalObj.append({"id": document.get('_id'), "key": "4", "label": document.get('modal'), "files": document.get('files')})
    return jsonify({"key": "4", "label": document.get('modal'), "files": document.get('files')})

@app.route('/model/<modalId>', methods=['DELETE'])
def delete_model(modalId):
    db = client['vector_files']
    file_collection = db["file_collection"]
    object_id = ObjectId(modalId)
    result = file_collection.delete_one({"_id": object_id})
    # Check if the document was deleted
    if result.deleted_count > 0:
        print("Document deleted successfully.", result)
        return "Document deleted successfully."
    else:
        print("No document found with the specified _id.", result)
        return "No document found with the specified _id."
    
@app.route('/model/file/<fileName>/id/<modalId>', methods=['DELETE'])
def delete_model_files(fileName, modalId):
    db = client['vector_files']
    file_collection = db["file_collection"]
    object_id = ObjectId(modalId)

    # Use $pull to remove the filename from the files array
    result = file_collection.update_one(
        {"_id": object_id},
        {"$pull": {"files": fileName}}
    )
    
    # Check if the filename was removed
    if result.modified_count > 0:
        print("Filename removed successfully.", result)
        return "Filename removed successfully."
    else:
        print("No document found with the specified _id and filename.", result)
        return "No document found with the specified _id and filename."

@app.route('/saved_modals', methods=['GET'])
def fetch_modals():
    db = client['vector_files']
    file_collection = db["file_collection"]
    cursor = file_collection.find({})

    modalObj = []
    for document in cursor:
        modalObj.append({"id": str(document.get('_id')), "key": "4", "label": document.get('modal'), "files": document.get('files')})
    return jsonify(modalObj)

@app.route('/test', methods=['GET'])
def _test():
    db = initialize_chroma_db('./data')
    print(db.get().keys())
    print(set([doc['source'] for doc in db.get()['metadatas']]))
    
    return "HI"

@app.route('/upload/<modalName>', methods=['POST'])
def upload_files(modalName):
    db = client['vector_files']
    file_collection = db["file_collection"]

    files = request.files.getlist('files')  # 여러 파일을 수신할 수 있도록 getlist 사용

    if not files:
        return jsonify({'error': 'No selected files'}), 400

    saved_files = []
    for file in files:
        original_filename = secure_filename(file.filename)

        # 원래 파일 이름을 사용하여 저장
        unique_filename = f"{random.randint(10000, 99999)}_{original_filename}"


        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        saved_files.append(unique_filename)

    # 몽고db insert
    files = {"modal": modalName, "files": saved_files }
    file_collection.insert_one(files)

    return jsonify({'message': 'Files successfully uploaded', 'files': saved_files}), 200


if __name__ == '__main__':
    api_key = load_api_key()
    set_api_key_env_var(api_key)
    app.run(host='0.0.0.0', port=5001, threaded=True, debug=True)
