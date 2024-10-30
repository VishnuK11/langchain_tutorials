import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    docs = file_loader.load()
    return docs

def chunk_data(docs, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc

def embeddings():
    embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
    return  embeddings

def vector_db(docs):
    index_name = "sample-db"
    # vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings())
    vector_store = PineconeVectorStore.from_documents(documents=docs,  embedding=embeddings(), index_name=index_name)
    # vector_store.similarity_search_with_score()
    return vector_store

def retrieve(query, vector_db, k=3):
    matching_results = vector_db.similarity_search_with_score(query, k=k)
    return matching_results

def context_prompt(context, query):
    template = """
    Based only on the below context, please answer the question below. 
    
    {context}
    
    {query}
    """
    llm = ChatOpenAI(model='gpt-4o')
    prompt = ChatPromptTemplate.from_template(template=template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({'context':context,'query':query})
    print()
    print(response)

if __name__ == '__main__':
    doc = read_doc("C:/Users/User.ET-A00186.000/Downloads/")
    # print(doc)
    chunk_docs = chunk_data(doc)
    vector_index = vector_db(chunk_docs)
    # print(chunk_docs[0:10])
    query = "I have lost the share certificate. What should I do?"
    query_result = retrieve(query=query, vector_db=vector_index )
    print()
    print(query)
    print(query_result)
    context = ""
    for doc,score, in query_result:
        context += doc.page_content
    context_prompt(context,query)



