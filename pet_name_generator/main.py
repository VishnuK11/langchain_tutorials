import streamlit as st
from youtube_assistant import LangchainOpenAI
import textwrap

st.title("Youtube Assistant")
with st.sidebar:
    with st.form(key = 'my_form'):
        youtube_url = st.sidebar.text_area(
            label = "What is the YouTube video URL",
            max_chars=50
        )
        query = st.sidebar.text_area(label="Ask me about the video?",
                                     max_chars=50,
                                     key="query")

if query and youtube_url:
    db_chat = LangchainOpenAI(link=youtube_url)
    response, docs = db_chat.get_response_from_db(query=query)
    print(docs)
    st.subheader("Answer")
    st.text(textwrap.fill(response, width = 80))