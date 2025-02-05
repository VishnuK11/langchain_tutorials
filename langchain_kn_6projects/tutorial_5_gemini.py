import os
from langchain_google_genai.llms import GoogleGenerativeAI
from dotenv import load_dotenv
from markdown import markdown

load_dotenv(dotenv_path=".env")


chat = GoogleGenerativeAI(api_key=os.getenv('GEMINI_API_KEY'), model='gemini-pro')

print(markdown(chat.invoke("What is the purpose of life")))