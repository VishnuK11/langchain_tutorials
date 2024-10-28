# Tutorial asks me to create a db in Cassandra-AstraDB

from langchain_community.vectorstores.cassandra import Cassandra
from langchain_astradb import AstraDBVectorStore
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv
# from dataset import load_dataset
from typing_extensions import Concatenate
import cassio
import os

load_dotenv()

class Pdf_RAG:
    def __init__(self):
        self.embedding = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
        self.llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-4o')
        self.raw_text = self.read_pdf()
        self.create_db()
        self.docs = self.get_docs()
        self.vector_store.add_texts(self.docs)


    def read_pdf(self):
        raw_text = ''
        page_limit = 4
        pdf_reader = PdfReader('C:/Users/User.ET-A00186.000/Downloads/FAQ_airtel.pdf')
        for i,page in enumerate(pdf_reader.pages):
            content = page.extract_text()
            if content:
                raw_text += content
            # if i>=page_limit:
            #     break
        # print(raw_text)
        return raw_text


    def create_db(self):
        # cassio.init(token=os.getenv('ASTRA_DB_APPLICATION_TOKEN'), database_id=os.getenv('ASTRA_DB_API_ENDPOINT'))
        self.vector_store = AstraDBVectorStore(collection_name='test', embedding=self.embedding,
                           token=os.getenv('ASTRA_DB_APPLICATION_TOKEN'), api_endpoint=os.getenv('ASTRA_DB_API_ENDPOINT'),
                           namespace=os.getenv('ASTRA_DB_KEYSPACE')
                           )

        # self.vector_store = Cassandra(embedding=self.embedding, table_name='vector-db', session=None, keyspace=None)
        self.vector_index = VectorStoreIndexWrapper(vectorstore=self.vector_store)

    def get_docs(self):
        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50, separator="\n")
        docs = splitter.split_text(self.raw_text)
        return docs

    def run_qns(self, qn_no=1):
        questions = {1:"I have not received dividend for past years, how can I claim the dividend?",
                     2:"Can you explain the pre-training method used for BioGPT"}
        response = self.vector_index.query(questions[qn_no], llm=self.llm).strip()

        print(response)
        response = self.vector_store.similarity_search(questions[qn_no], k=4)
        print(response)


if __name__ == '__main__':

    read = Pdf_RAG()
    read.run_qns()
