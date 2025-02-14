# LangChain Components

## Introduction

Large Language Models (LLMs) have high computational requirements, making it difficult to run them locally. Companies like OpenAI and Anthropic provide APIs for developers to use their models via the cloud. However, different companies have different API syntax, creating a standardization problem. For example, switching from OpenAI's GPT-4.0 to Claude 3.5 Sonnet requires changing the entire codebase.

LangChain simplifies this by unifying the interface, so switching LLMs only requires changing two lines of code:
1. When importing the LLM
2. When calling the model

LangChain provides two key model types:
- **Language Models**
- **Embedding Models**

## 1. Prompts

The output of an LLM depends entirely on the input **prompt**. Even a small change in wording can drastically alter the response.

### Dynamic and Usable Prompts

We can make prompts reusable by inserting placeholders, which can be dynamically filled by the user.

#### Example:
```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["name", "topic"],
    template="Hi {name}, can you explain {topic} to me?"
)

print(prompt.format(name="Rahul", topic="Neural Networks"))
```

### Role-based Prompts

LLMs can be instructed to assume a specific role, making responses more accurate and domain-specific.

#### Example:
```python
from langchain.prompts import PromptTemplate

role_based_prompt = PromptTemplate(
    input_variables=["profession", "topic"],
    template="You are an experienced {profession}. Explain {topic} in simple terms."
)

print(role_based_prompt.format(profession="Electrical Engineer", topic="Circuits"))
```

### Few-shot Prompting

Few-shot prompting provides examples to guide the LLM in generating better responses.

#### Example:
```python
from langchain.prompts import FewShotPromptTemplate

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "3+5", "output": "8"}
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Q: {input}\nA: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Solve the following math problem:",
    suffix="Q: {query}\nA:",
    input_variables=["query"]
)

print(few_shot_prompt.format(query="7+3"))
```

## 2. Chains

Chains act as pipelines, passing outputs of one step to the next. Without chains, you would have to handle these steps manually.

### Example: Translating and Summarizing
```python
from langchain.chains import SimpleSequentialChain
from langchain.llms import OpenAI

llm = OpenAI()

translate_chain = LLMChain(llm=llm, prompt=PromptTemplate(template="Translate this to Nepali: {text}", input_variables=["text"]))
summarize_chain = LLMChain(llm=llm, prompt=PromptTemplate(template="Summarize this: {text}", input_variables=["text"]))

full_chain = SimpleSequentialChain(chains=[translate_chain, summarize_chain])

result = full_chain.run("LangChain is a great tool for LLMs.")
print(result)
```

### Complex Chains:
- **Parallel Chains:** Combine outputs from multiple LLMs before presenting them to the user.
- **Conditional Chains:** Execute different chains based on user input.

## 3. Indexes

Indexes help LLMs retrieve relevant information efficiently. They are useful for searching large datasets or knowledge bases.

## 4. Memory

LLMs are stateless, meaning they do not remember previous interactions. Memory modules help maintain context across API calls.

### Example:
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "Who is Prachanda?"}, {"output": "He is a Nepalese politician."})

print(memory.load_memory_variables({}))
```

## 5. Agents

Agents allow LLMs to dynamically decide which tools or functions to use based on user queries.

### Example:
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
LangChain makes working with LLMs easier by:
1. **Standardizing** LLM integration across different providers.
2. **Optimizing** prompt engineering techniques.
3. **Enabling** complex multi-step workflows through chains.
4. **Maintaining** memory across user interactions.
5. **Empowering** LLMs with tool-based decision-making using agents.

This allows ML engineers to focus on building intelligent applications without worrying about the complexities of handling different LLM APIs manually.

