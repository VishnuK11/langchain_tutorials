import os

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



class LangchainOpenAI:
    def __init__(self, link):
        self.llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.loader = YoutubeLoader.from_youtube_url(link)
        self.transcript = self.loader.load()
        print(self.transcript)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.docs = self.splitter.split_documents(self.transcript)
        embeddings = OpenAIEmbeddings()
        self.db = FAISS.from_documents(self.docs, embeddings)
        self.retrieved_docs = None

    def get_response_from_db(self, query):
        self.retrieved_docs = self.db.similarity_search(query,k=3)
        docs_page_content = " ".join([d.page_content for d in self.retrieved_docs])
        self.prompt_template = ChatPromptTemplate.from_messages(["""
                                Based on the following context, answer the following question
                                
                                {docs}
                                
                                {question}
                                
        """])
        self.db.as_retriever()
        chain = self.prompt_template | self.llm | StrOutputParser()
        response = chain.invoke({"docs": docs_page_content, "question": query})
        return response, docs_page_content


if __name__ == '__main__':
    chat = LangchainOpenAI('https://www.youtube.com/watch?v=lG7Uxts9SXs')
    print(chat.get_response_from_db(query='Get details of Ransomware spoken in the video'))
