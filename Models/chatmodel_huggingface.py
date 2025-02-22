from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not hf_token:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found! Check your .env file.")

print("Hugging Face Token Loaded:", hf_token[:5] + "..." + hf_token[-5:])

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of Nepal?")

print(result.content)
