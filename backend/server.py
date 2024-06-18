import os, json, random, configparser,threading, queue
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import GPT4All
from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from werkzeug.utils import secure_filename
from langchain.chains import RetrievalQA, create_retrieval_chain, create_history_aware_retriever, RetrievalQAWithSourcesChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.utils import initialize_chroma_db
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from guardrails.hub import RegexMatch, TwoWords
from guardrails import Guard, OnFailAction
from guardrails.errors import ValidationError
from pydantic import BaseModel
import litellm

from bson import ObjectId
import warnings
warnings.filterwarnings("ignore")

load_dotenv('.env.development')
app = Flask(__name__)

MONGODB_URL = os.getenv("MONGODB_URL")
client = MongoClient(MONGODB_URL)
CORS(app)
UPLOAD_FOLDER = 'data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def set_api_key_env_var():
    config = configparser.ConfigParser()
    config.read('config.ini')
    api_key = config['API']['OPENAI_API_KEY']
    huggingface_key = config['HUGGINGFACE_API']['HUGGINGFACEHUB_API_TOKEN']
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_key
    

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

def _get_chat_history(session_id):
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGODB_URL,
        database_name="chat_db",
        collection_name="chat_histories",
    )

def llm_OpenAI(g, prompt, sessionId):
    try:
        model = ChatOpenAI(
            verbose=True,
            streaming=True,
            callbacks=[ChainStreamHandler(g)],
            temperature=0.7,
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        llm_chain = (prompt_template | model | StrOutputParser())
        # response = completion(
        #     model="ollama/llama2",
        #     messages = [{ "content": "Hello, how are you?","role": "user"}],
        #     api_base="http://localhost:11434",
        #     stream=True,
        # )

        # Create the RunnableWithMessageHistory instance
        chain_with_history = RunnableWithMessageHistory(
            llm_chain,
            _get_chat_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        config = {"configurable": {"session_id": sessionId}}    
        _res = chain_with_history.invoke({"question": prompt}, config=config) 
    finally:
        g.close()

def llm_gpt4(g, prompt, sessionId):
    try:
        local_path = './models/nous-hermes-llama2-13b.Q4_0.gguf'
        model = GPT4All(
            model=local_path,
            callbacks=[ChainStreamHandler(g)],
            streaming=True,
            verbose=True,
        )
        # llm_chain.invoke({"text": ""})
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        llm_chain = prompt_template | model

        # Create the RunnableWithMessageHistory instance
        chain_with_history = RunnableWithMessageHistory(
            llm_chain,
            _get_chat_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        config = {"configurable": {"session_id": sessionId}}    
        _res = chain_with_history.invoke({"question": prompt}, config=config)         
    finally:
        g.close()

def llm_Ollama(g, prompt, sessionId):
    try:
        # model = Ollama(
        #     model='platypus-kor',
        #     callbacks=[ChainStreamHandler(g)],
        #     verbose=True,
        # )    
        
        # prompt_template = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", "You are a helpful assistant."),
        #         MessagesPlaceholder(variable_name="history"),
        #         ("human", "{question}"),
        #     ]
        # )
        # llm_chain = prompt_template | model

        # Call the Guard to wrap the LLM API call
        class Pet(BaseModel):
            name: str
            age: int


        guard = Guard.from_pydantic(Pet)
        guard.use(TwoWords, on=prompt)
        # guard.with_prompt_validation([TwoWords(on_fail="exception")])

        raw_llm_response, validated_response, *rest = guard(
            litellm.completion,
            model="ollama/llama2",
            max_tokens=1000,
            api_base="http://localhost:11434",
            stream=True,
        )
        # response = litellm.completion(
        #     model="ollama/llama2", 
        #     messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
        #     msg_history=[{"role": "user", "content": "hello"}],
        #     api_base="http://localhost:11434",
        #     stream=True
        # )
        print(raw_llm_response)
        print(validated_response)
        print(*rest)
        # for chunk in response:
        #     print(chunk['choices'][0]['delta'])        

        # print('validated_response', validated_response)
        
        # Create the RunnableWithMessageHistory instance
        # chain_with_history = RunnableWithMessageHistory(
        #     llm_chain,
        #     _get_chat_history,
        #     input_messages_key="question",
        #     history_messages_key="history",
        # )

        # config = {"configurable": {"session_id": sessionId}}    
        # _res = chain_with_history.invoke({"question": prompt}, config=config) 
    finally:
        g.close()        

def llm_openAI_with_chroma(g, prompt, sessionId):
    # 학습된 pdf 파일명, 경로에 따라 학습할지 말지를 결정
    # 학습이 안된 파일이 있다면, 잠시 시간이 걸리면서 vector로 변환
    db = initialize_chroma_db('./data')
    # 로드된 DB를 이용하여 Retriever 초기화
    model = ChatOpenAI(
            verbose=True,
            streaming=True,
            callbacks=[ChainStreamHandler(g)],
            temperature=0.7,
        )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    # retriever_chain = create_history_aware_retriever(model, db.as_retriever(), prompt_template)
    # rag_chain = create_retrieval_chain(db.as_retriever(), create_stuff_documents_chain(model, prompt_template))
    # chain = RetrievalQA.from_chain_type(
    #     llm=model,
    #     chain_type="stuff",
    #     retriever=db.as_retriever(),
    #     return_source_documents=True,
    # )
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=db.as_retriever(),
        # prompt=prompt_template
    )
    # question_answer_chain = create_stuff_documents_chain(model, prompt_template)
    # chain = create_retrieval_chain(db.as_retriever(), question_answer_chain)

    # # Create the RunnableWithMessageHistory instance
    # chain_with_history = RunnableWithMessageHistory(
    #     chain,
    #     _get_chat_history,
    #     input_messages_key="question",
    #     history_messages_key="history",
    # )

    config = {"configurable": {"session_id": sessionId}}    
    _res = chain.invoke({"question": prompt}, config=config)     
    # _res = chain.invoke(prompt)
    g.close()

def chain(prompt, modal, sessionId):
    g = ThreadedGenerator()
    if modal == '1':
        threading.Thread(target=llm_OpenAI, args=(g, prompt, sessionId)).start()
    elif modal == '2':
        threading.Thread(target=llm_gpt4, args=(g, prompt, sessionId)).start()
    elif modal == '3':
        threading.Thread(target=llm_Ollama, args=(g, prompt, sessionId)).start()
    else:
        threading.Thread(target=llm_openAI_with_chroma, args=(g, prompt, sessionId)).start()

    return g

@app.route('/generate/<modal>/id/<sessionId>', methods=['POST'])
def _chain(modal, sessionId):
    return Response(chain(request.json['prompt'], modal, sessionId), mimetype='text/plain')     # OK

@app.route('/model/<modalId>', methods=['GET'])
def edit_modals(modalId):
    db = client['vector_files']
    file_collection = db["file_collection"]
    object_id = ObjectId(modalId)
    document = file_collection.find_one({"_id": object_id})

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

@app.route('/initiate', methods=['GET'])
def _initiate():
    initialize_chroma_db('./data')
    return "Done"

@app.route('/chats', methods=['GET'])
def _fetchChatList():
    chat_lists = []
    seen_session_ids = set()
    db = client['chat_db']
    chat_histories = db['chat_histories']
    
    for chat_history in chat_histories.find():
        session_id = chat_history['SessionId']
        if session_id not in seen_session_ids:
            title = json.loads(chat_history["History"])["data"]["content"]
            chat_lists.append({"title": title, "sessionId": session_id})
            seen_session_ids.add(session_id)
    
    return chat_lists

@app.route('/history/<sessionId>', methods=['GET'])
def _fetchHistory(sessionId):
    history = []
    chat_message_history = _get_chat_history(sessionId).messages
    for idx, msg in enumerate(chat_message_history):
        if idx % 2 == 0:
            history.append({"text": [msg.content], "type": "user"})
        else:
            history.append({"text": [msg.content], "type": "bot"})
    return history

@app.route('/test', methods=['GET'])
def _test():
    chatLists = []
    db = client['chat_db']
    chat_histories = db["chat_histories"]
    for chat_history in chat_histories.find():
        if chat_history['SessionId'] not in chatLists:
            chatLists.append(chat_history['SessionId'])

    print(chatLists)
    return "good"


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
    set_api_key_env_var()
    app.run(host='0.0.0.0', port=5001, threaded=True, debug=True)
