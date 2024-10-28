import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import ZeroShotAgent, AgentType, load_tools, initialize_agent



class LangchainOpenAI:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def llm_chat(self):
        response = self.llm.invoke("Can you suggest me names for a dog that is grey in color?")
        # parser = StrOutputParser()
        return response

    def llm_chat_chain(self):
        # response = self.llm.invoke("")
        parser = StrOutputParser()
        chain = self.llm | parser
        response = chain.invoke(input="Can you suggest me names for a dog that is grey in color?",
                                config={"configurable": {"output_key": "pet_name"}})
        return response

    def llm_chat_prompt_template(self):
        prompt = ChatPromptTemplate.from_template(template="Can you suggest me names for a {pet} that is {color} in color?")
        parser = StrOutputParser()
        chain =  prompt | self.llm | parser
        response = chain.invoke({"pet": "dog", "color": "yellow"})
        return response


    def llm_chat_agent(self):
        llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o")
        tools = load_tools(["wikipedia"])
        agent = initialize_agent(tools=tools, llm=llm, AgentType=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        response = agent.run("Find the average lifespan of dog. Multiple it by 5. Show each step of calculation and sources")
        return response


if __name__ == '__main__':
    chat = LangchainOpenAI()
    # ret = chat.llm_chat()
    # ret = chat.llm_chat_chain()
    # ret = chat.llm_chat_prompt_template()
    ret = chat.llm_chat_agent()
    print(ret)