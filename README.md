# LangChain

## Introduction

LangChain is a powerful framework designed to simplify the development of applications using large language models (LLMs) such as OpenAI's GPT, Anthropic's Claude, and Meta's Llama. It abstracts complex tasks like managing prompts, retrieving relevant data, integrating with APIs, and chaining multiple LLM calls, making it easier to build intelligent applications efficiently.

## Why LangChain?

### The Problem Before LangChain

Before LangChain, developers had to manually:
- Handle multiple API calls to different LLMs.
- Store and retrieve context-relevant documents.
- Chain multiple prompts and responses for complex workflows.
- Integrate AI models with external tools such as APIs and databases.

For ML engineers, using LLMs alongside traditional ML models required extensive custom logic for:
- Data retrieval (e.g., vector databases, embeddings).
- Context-aware decision-making.
- Memory handling for multi-turn conversations.

LangChain solves these problems by standardizing LLM interactions, making applications easier to scale and maintain.

## Getting Started

To install LangChain:
```bash
pip install langchain
```

For official documentation, visit [LangChain Documentation](https://python.langchain.com/).

---

# LangChain Components

LangChain is modular, allowing users to interact with different components independently. Below are the key components:

## [1. Models](Models/)

LLMs require a structured way to interact with different APIs. LangChain provides a unified interface, enabling developers to switch between models easily.

### Example: Using OpenAI GPT-4
```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
response = llm.predict("Explain the importance of AI in healthcare.")
print(response)
```

## [2. Prompts](Prompts/)

Prompts define the instructions given to LLMs, which significantly impact responses. LangChain allows developers to structure dynamic and reusable prompts.

### Example: Creating a Dynamic Prompt
```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["name", "topic"],
    template="Hi {name}, can you explain {topic} to me?"
)

print(prompt.format(name="Rahul", topic="Neural Networks"))
```

## [3. Chains](Chains/)

Chains help combine multiple steps in an AI workflow, passing outputs from one step to another.

### Example: Translating and Summarizing
```python
from langchain.chains import SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI()

translate_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template="Translate this to Nepali: {text}", input_variables=["text"])
)

summarize_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template="Summarize this: {text}", input_variables=["text"])
)

full_chain = SimpleSequentialChain(chains=[translate_chain, summarize_chain])
result = full_chain.run("LangChain is a great tool for LLMs.")
print(result)
```

## [4. Indexes](Indexes/)

Indexes allow LLMs to retrieve relevant information efficiently, especially when searching large datasets or knowledge bases.

### Example: Creating a Vector Store Index
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

texts = ["Machine learning is fascinating.", "AI is transforming the world."]
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(texts, embeddings)

query = "Tell me about AI."
results = vector_store.similarity_search(query)
for res in results:
    print(res.page_content)
```

## [5. Memory](Memory/)

LLMs are stateless by default. Memory components allow them to remember previous interactions, improving conversation flow.

### Example: Using Conversation Memory
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "Who is Prachanda?"}, {"output": "He is a Nepalese politician."})

print(memory.load_memory_variables({}))
```

## [6. Agents](Agents/)

Agents allow LLMs to dynamically select which tools to use based on user queries.

### Example: Using an Agent with a Custom Tool
```python
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool

def fetch_weather(location):
    return f"The weather in {location} is sunny."

weather_tool = Tool(
    name="WeatherAPI",
    func=fetch_weather,
    description="Provides weather updates."
)

agent = initialize_agent(
    tools=[weather_tool],
    llm=OpenAI(),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

print(agent.run("What is the weather in Kathmandu?"))
```

## Conclusion

LangChain simplifies working with LLMs by:
1. Standardizing LLM integration across providers.
2. Optimizing prompt engineering techniques.
3. Enabling multi-step workflows through chains.
4. Maintaining memory across user interactions.
5. Empowering LLMs with tool-based decision-making using agents.

By leveraging LangChain, developers can build powerful AI applications with minimal effort, focusing on high-level problem-solving instead of low-level API management.