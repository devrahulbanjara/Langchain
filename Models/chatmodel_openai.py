from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = "gpt-4")

model.invoke("What is the area of Nepal? ")
print(result)