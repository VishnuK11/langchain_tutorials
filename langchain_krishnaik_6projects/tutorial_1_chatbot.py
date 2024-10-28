import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import ZeroShotAgent, AgentType, initialize_agent
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage



class LangchainOpenAI:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=1)

    def llm_chat(self):
        response = self.llm.invoke("Can you suggest me names for a dog that is grey in color?")
        # parser = StrOutputParser()
        return response

    def llm_chat_chain(self):
        # response = self.llm.invoke("")
        parser = StrOutputParser()
        country = "Russia"
        chain = self.llm | parser
        response = chain.invoke(input=f"What is the capital of {country}",
                                config={"configurable": {"output_key": "capital"}})
        return response

    def llm_chat_prompt_template(self):
        capital_prompt = ChatPromptTemplate.from_template(template="What is the capital of {country}")
        parser = StrOutputParser()
        chain = capital_prompt | self.llm | parser
        response = chain.invoke({"country": "Russia"})
        return response

    def llm_chat_sequential(self):
        capital_prompt = ChatPromptTemplate.from_template(template="What is the capital of the {country}", output_key='capital')
        parser = StrOutputParser()
        famous_prompt = ChatPromptTemplate.from_template(template="""Can you suggest me top 3 places to visit in {capital}. 
        Share the output in the format {{country:<country_name>, capital: <capital_name>, places:<list_of_places>}}""", output_key='places')
        chain_pre = capital_prompt | self.llm | parser
        chain = {"capital":chain_pre} | famous_prompt | self.llm | parser
        response = chain.invoke({"country": "India"})
        return response

    def llm_chat_message(self):
        response = self.llm.invoke([
            SystemMessage(content="You are a comedian AI assistant"),
            HumanMessage(content="Tell a funny joke about ML and LLM that is 1-2 lines long")
        ])
        return response.content


    def llm_chat_agent(self):
        llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o")
        tools = load_tools(["wikipedia"])
        agent = initialize_agent(tools=tools, llm=llm, AgentType=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        response = agent.run("Find the average lifespan of dog. Multiple it by 5. Show each step of calculation and sources")
        return response

    def custom_parser(self):
        template = "You are a helpful assistant. Generate 5 synonyms for a word provided by user. "
        human_template = "{text}"
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system",template),
            ("user",human_template)
        ])
        chain = chat_prompt | self.llm | CSV_Output()
        response = chain.invoke({"text":"intelligent"})
        return response


class CSV_Output(BaseOutputParser):
    def parse(self, text:str):
        return text.strip().split("\n")

if __name__ == '__main__':
    chat = LangchainOpenAI()
    # ret = chat.llm_chat()
    # ret = chat.llm_chat_chain()
    # ret = chat.llm_chat_prompt_template()
    # ret = chat.llm_chat_sequential()
    # ret = chat.llm_chat_agent()
    ret = chat.llm_chat_message()
    ret = chat.custom_parser()
    print(ret)