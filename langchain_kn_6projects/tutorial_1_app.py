# Q&A ChatBot

from dotenv import load_dotenv
import streamlit as st
from tutorial_1_chatbot import LangchainOpenAI
import os

load_dotenv()


class App:
    def __init__(self):
        self.cb = LangchainOpenAI()
        self.response =None
    def run(self):
        self.response = self.cb.llm_chat_sequential()
        return self.response


if __name__ == '__main__':
    st.title("Langchain Application: Q&A Demo")
    st.text("What is your question?")
    text = st.text_area(label='input',max_chars=100)
    submit = st.button("Generate Answer")

    if submit:
        output = App()
        st.text(output.run())



