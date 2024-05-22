# app.py  파일
# 실행법 -> streamlit run app.py
import streamlit as st
import requests

# LangChain 서버 URL
llm_server_url = "http://localhost:5000/generate"

st.title("GPT4All 모델 기반 LLM 웹 UI")

# 사용자 입력
user_input = st.text_area("질문을 입력하세요:")
# print("user_input", user_input)
if st.button("전송"):
    if user_input:
        response = requests.post(llm_server_url, json={"text": user_input})
        if response.status_code == 200:
            st.write("응답: ", response.json().get('response'))
        else:
            st.write("오류 발생: ", response.text)
    else:
        st.write("질문을 입력해주세요.")

if st.button("초기화"):
    st.text_area("질문을 입력하세요:", value="", key="reset")
