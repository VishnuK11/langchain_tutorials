from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import os
import re


load_dotenv()


# Get a specific environment variable
def is_valid_api_key():
    api_key = os.environ.get('OPENAI_API_KEY')
    pattern = r'^sk-proj-[A-Za-z0-9_-]{156}$'
    if not (bool(re.match(pattern, api_key))):
        raise Exception("OPEN AI String Not working")

def test_llm():
    llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = llm.invoke("Who is the Prime Minister of India in 2005 and 2015")
    print(f"{response.content}")

def test_prompt_template():
    llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    prompt_template = ChatPromptTemplate.from_messages([("system","You are a career advisor helping candidates improve their skills"),
                                                        ("user","{input_msg}")])
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser
    response = chain.invoke({"input_msg": "How to be an advanced Large Language Model (LLM) and GEN-AI engineer"})
    print(response)


# def test_hf_neo():
#     llm = HuggingFaceHub(model)


def test_web_loader():
    llm = ChatOpenAI()
    loader = WebBaseLoader("https://raghunaathan.medium.com/query-translation-for-rag-retrieval-augmented-generation-applications-46d74bff8f07")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embedding=OpenAIEmbeddings())

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following questions based on the context below
        
        <context>
        {context}
        </context>
        
        Question: {input}
"""
    )

    documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, documents_chain)
    response = retrieval_chain.invoke({"input": "What is query decomposition"})
    print(response)

def test_retrieval_chain():
    llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/LangChain")

    docs = loader.load()
    prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a summary of the conversation")
])
    vector = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())
    retriever = vector.as_retriever()
    chat_history = [HumanMessage(content="Has Langchain raised funding?"), AIMessage(content="Yes!")]
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    response = retriever_chain.invoke({
        "chat_history": chat_history,
        "input": "Can you tell me who funded langchain and how much"
    })

    print(response)
    # document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    # document_chain.invoke({
    #     "input": "how can langsmith help with testing?",
    #     "context": [Document(page_content="langsmith can let you visualize test results")]
    # })


def test_retrieval_chain_2():

    # First we need a prompt that we can pass into an LLM to generate this search query
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    docs = loader.load()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    retriever = vector.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    response = retriever_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })

    print(response)

if __name__ == '__main__':
    is_valid_api_key()
    # test_prompt_template()
    # test_web_loader()
    test_retrieval_chain_2()