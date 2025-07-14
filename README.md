# AI-agent-building


Here’s a **Beginner to Advanced Roadmap** to learn and build **AI Agents** step by step:

---

## 🧠 **1. Fundamentals (Beginner)**

### ✅ Learn Python

* Variables, loops, functions, classes
* Libraries: `requests`, `json`, `os`, `dotenv`

### ✅ Understand LLM Basics

* What is a Large Language Model?
* Key terms: token, context window, fine-tuning, embeddings

### ✅ Tools You Need

* OpenAI API (GPT-4, GPT-3.5)
* Jupyter or VSCode
* Python environment: `venv`, `pip`, `conda`

---

## 🛠️ **2. LLM APIs + Prompting (Core Agent Skills)**

### ✅ Prompt Engineering

* Zero-shot, few-shot, chain-of-thought prompting
* System prompt vs user prompt

### ✅ Use OpenAI / Anthropic APIs

* `openai.ChatCompletion.create()`
* Role-based conversations (system, user, assistant)

---

## ⚙️ **3. Build First AI Agents (Using LangChain)**

### ✅ LangChain Basics

* Chains: SequentialChain, SimpleSequentialChain
* Memory: ConversationBufferMemory
* Tools & Agents: ReAct, ToolAgent

### ✅ Add Tools to Agents

* Calculator, Search, Python REPL, API calls

---

## 🔗 **4. Integrate with External APIs**

* Web scraping with `requests`, `BeautifulSoup`, `playwright`
* Call external APIs (weather, news, etc.)
* LangChain’s `RequestsTool`, `SerpAPIWrapper`

---

## 🗂️ **5. Vector Stores & RAG (Retrieval Augmented Generation)**

### ✅ Index Knowledge

* Use FAISS / Chroma / Weaviate
* Store and query embeddings

### ✅ Implement RAG

* Upload PDF / Docs → chunk → embed → query → answer

---

## 🤖 **6. Multi-Agent Systems (Advanced)**

* Agents that delegate tasks
* Build workflows: planner → executor
* Use **CrewAI**, **Autogen**, or **LangGraph**

---

## 🔐 **7. Memory, State, and Tools**

* Add context-aware memory (Long-term / Episodic)
* Tool calling with OpenAI functions
* Agents that remember previous tasks

---

## 🧪 **8. Deploy & Automate**

* Deploy with **FastAPI / Streamlit**
* Use **Docker** for containerization
* Use **Celery + Redis** for async task queues

---

## 🧠 Advanced Projects

### 🔹 Personal Assistant Agent

* Query calendar, email, files

### 🔹 Research Assistant Agent

* Search + summarize + cite

### 🔹 Coding Assistant Agent

* Use ReAct + Python tool

---

## 🔗 Bonus Libraries

* `LangChain`, `LlamaIndex`, `CrewAI`, `AutoGen`, `Semantic Kernel`
* `FAISS`, `Chroma`, `Weaviate` (for RAG)

Great! Here's a deeper breakdown for:

---

## 🧠 **1. Fundamentals (Beginner)**

### ✅ **Learn Python**

#### 📌 **1.1 Core Concepts**

* **Variables** – Store data

  ```python
  name = "AgentX"
  age = 3
  ```

* **Loops** – Repeat tasks

  ```python
  for i in range(5):
      print("Loop", i)
  ```

* **Functions** – Encapsulate logic

  ```python
  def greet(name):
      return f"Hello, {name}"
  ```

* **Classes** – OOP for building agents

  ```python
  class Agent:
      def __init__(self, name):
          self.name = name
      def run(self):
          print(f"{self.name} is thinking...")

  agent = Agent("Astra")
  agent.run()
  ```

---

#### 📌 **1.2 Must-Know Libraries**

| Library    | Purpose                                |
| ---------- | -------------------------------------- |
| `requests` | Make HTTP API calls                    |
| `json`     | Parse/format JSON                      |
| `os`       | Access environment and system          |
| `dotenv`   | Load environment variables from `.env` |

**Example:**

```python
import requests, json, os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hi"}]}
)
print(response.json())
```

Perfect! Here's the next step:

---

## ⚙️ **2. LLM APIs + Prompting (Core Agent Skills)**

### ✅ **2.1 Prompt Engineering**

Understanding how to talk to LLMs effectively is the **core skill**.

#### 🔹 Prompt Types:

| Prompt Type      | Example                      |
| ---------------- | ---------------------------- |
| Zero-shot        | “Translate to French: Hello” |
| Few-shot         | “Q: 2+2? A: 4\nQ: 3+3? A:”   |
| Chain-of-Thought | “Let’s think step by step…”  |

---

### ✅ **2.2 OpenAI API Integration**

Install the library:

```bash
pip install openai
```

#### 🔹 Basic Example with ChatCompletion:

```python
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful AI agent."},
    {"role": "user", "content": "What's the capital of Japan?"}
  ]
)

print(response['choices'][0]['message']['content'])
```

---

### ✅ **2.3 Roles in Chat Messages**

| Role        | Purpose                             |
| ----------- | ----------------------------------- |
| `system`    | Sets the behavior of the assistant  |
| `user`      | The human input                     |
| `assistant` | (optional) model’s previous outputs |

---

### ✅ **2.4 Prompt Design Tips**

* Give **context**
* Use **examples**
* Be **explicit** about expected output

**Example:**

```python
messages=[
  {"role": "system", "content": "You are a JSON-generating assistant."},
  {"role": "user", "content": "Give me a JSON of a car with make, model, and year"}
]
```

Awesome! Let's move to the **agent-building phase** with LangChain:

---

## 🤖 **3. Build First AI Agents (Using LangChain)**

### ✅ **3.1 What is LangChain?**

LangChain helps you build **AI agents** by chaining LLMs + tools + memory + logic.

Install it:

```bash
pip install langchain openai
```

---

### ✅ **3.2 Basic LangChain Chat**

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(model="gpt-3.5-turbo")
response = llm([HumanMessage(content="Tell me a joke")])
print(response.content)
```

---

### ✅ **3.3 Chains**

* **SimpleSequentialChain** – LLM → Output → LLM
* **LLMChain** – LLM with prompt template

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate.from_template(template)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("robot pets"))
```

---

### ✅ **3.4 Add Memory to Agent**

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
chat_with_memory = ConversationChain(llm=llm, memory=memory)

chat_with_memory.predict(input="Hi!")
chat_with_memory.predict(input="What did I just say?")
```

---

### ✅ **3.5 Agent with Tools (Tool-Using Agent)**

```python
from langchain.agents import initialize_agent, load_tools

tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("What is the square root of 2025 plus current weather in Delhi?")
```

Great! Let's now move into **Tool Integration** – giving your agent superpowers beyond text:

---

## 🔗 **4. Integrate with External APIs (APIs, Web, Files)**

### ✅ **4.1 Call External APIs (Requests Tool)**

```python
import requests

url = "https://api.weatherapi.com/v1/current.json"
params = {"key": "your_api_key", "q": "Bangalore"}
res = requests.get(url, params=params)
print(res.json())
```

🔹 Use this output in an agent via a custom tool in LangChain.

---

### ✅ **4.2 Web Scraping with Playwright or BeautifulSoup**

```python
from bs4 import BeautifulSoup
import requests

res = requests.get("https://news.ycombinator.com/")
soup = BeautifulSoup(res.text, "html.parser")

for item in soup.select(".titleline > a")[:5]:
    print(item.text)
```

---

### ✅ **4.3 LangChain Tool Integration Example**

```python
from langchain.tools import Tool

def get_news(query):
    return f"Search results for: {query}"

custom_tool = Tool(
    name="CustomNewsSearch",
    func=get_news,
    description="Useful for searching news"
)

from langchain.agents import initialize_agent
agent = initialize_agent([custom_tool], llm, agent="zero-shot-react-description", verbose=True)
agent.run("Search news about OpenAI")
```

---

### ✅ **4.4 File I/O – Upload & Read Files**

```python
with open("myfile.txt", "r") as f:
    content = f.read()
print(content)
```

Use with LangChain’s document loaders:

```python
from langchain.document_loaders import TextLoader
loader = TextLoader("myfile.txt")
docs = loader.load()
```
Great! Let's now move into **Tool Integration** – giving your agent superpowers beyond text:

---

## 🔗 **4. Integrate with External APIs (APIs, Web, Files)**

### ✅ **4.1 Call External APIs (Requests Tool)**

```python
import requests

url = "https://api.weatherapi.com/v1/current.json"
params = {"key": "your_api_key", "q": "Bangalore"}
res = requests.get(url, params=params)
print(res.json())
```

🔹 Use this output in an agent via a custom tool in LangChain.

---

### ✅ **4.2 Web Scraping with Playwright or BeautifulSoup**

```python
from bs4 import BeautifulSoup
import requests

res = requests.get("https://news.ycombinator.com/")
soup = BeautifulSoup(res.text, "html.parser")

for item in soup.select(".titleline > a")[:5]:
    print(item.text)
```

---

### ✅ **4.3 LangChain Tool Integration Example**

```python
from langchain.tools import Tool

def get_news(query):
    return f"Search results for: {query}"

custom_tool = Tool(
    name="CustomNewsSearch",
    func=get_news,
    description="Useful for searching news"
)

from langchain.agents import initialize_agent
agent = initialize_agent([custom_tool], llm, agent="zero-shot-react-description", verbose=True)
agent.run("Search news about OpenAI")
```

---

### ✅ **4.4 File I/O – Upload & Read Files**

```python
with open("myfile.txt", "r") as f:
    content = f.read()
print(content)
```

Use with LangChain’s document loaders:

```python
from langchain.document_loaders import TextLoader
loader = TextLoader("myfile.txt")
docs = loader.load()
```
Perfect! Let's now level up with **RAG (Retrieval-Augmented Generation)** – where your agent learns from **documents** like PDFs, Notion, websites, and more.

---

## 📚 **5. RAG (Retrieval-Augmented Generation)**

### ✅ **What is RAG?**

**RAG = Embed + Store + Retrieve + Answer**

> Combine LLM + vector DB to answer questions over custom data (PDFs, Docs, etc.)

---

### ✅ **5.1 Install Required Packages**

```bash
pip install langchain faiss-cpu tiktoken
pip install chromadb # (optional: for Chroma DB)
pip install pypdf
```

---

### ✅ **5.2 Load and Split Documents**

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("myfile.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)
```

---

### ✅ **5.3 Generate Embeddings + Store in FAISS**

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Save to disk
vectorstore.save_local("my_faiss_index")
```

---

### ✅ **5.4 Ask Questions from Your Docs (RAG)**

```python
from langchain.chains import RetrievalQA
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
response = qa_chain.run("What are the key topics in the PDF?")
print(response)
```

---

### ✅ **5.5 Optional: Use Chroma or Weaviate Instead of FAISS**

* `Chroma` = local storage
* `Weaviate` = scalable, cloud or local
* `Pinecone` = managed vector DB (cloud)

Awesome! Let’s now explore how to build **Multi-Agent Systems** — where multiple AI agents collaborate to complete complex tasks.

---

## 🤖 **6. Multi-Agent Systems (Advanced)**

> Agents that **plan, delegate, and execute** — like human teams.

---

### ✅ **6.1 CrewAI – Team of Agents**

**Install:**

```bash
pip install crewai
```

**Example:**

```python
from crewai import Crew, Agent, Task
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

researcher = Agent(
    role="Researcher",
    goal="Find latest news on AI",
    backstory="AI news analyst",
    llm=llm
)

writer = Agent(
    role="Writer",
    goal="Summarize AI news",
    backstory="Technical writer",
    llm=llm
)

task1 = Task(agent=researcher, description="Search for top 3 AI breakthroughs in 2024.")
task2 = Task(agent=writer, description="Summarize those findings in plain English.")

crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
crew.kickoff()
```

---

### ✅ **6.2 LangGraph – Agent State Machines (Open Source)**

```bash
pip install langgraph
```

* Define states (e.g., `Plan`, `Execute`, `Summarize`)
* Powerful for **agent memory**, loops, conditional logic

**Use-case:** data pipelines, agent workflows

---

### ✅ **6.3 AutoGen – Advanced LLM-to-LLM Communication**

```bash
pip install pyautogen
```

* Supports chat between agents
* Agents can be code executors, planners, assistants

**Example:**

* User → Planner Agent → Executor Agent (code runs) → Refiner Agent

---

### ✅ Agent Design Patterns:

| Pattern              | Description                          |
| -------------------- | ------------------------------------ |
| **Planner/Executor** | Plan tasks, delegate to sub-agents   |
| **Chain of Agents**  | Output of one = Input of next        |
| **Voting Agents**    | Multiple agents propose, one decides |
| **Recursive Agents** | Agents that self-correct             |

Great! Let’s now explore how to give your AI agents **memory**, maintain **state**, and **use tools** to act like intelligent assistants.

---

## 🔐 **7. Memory, State, and Tool Use**

---

### ✅ **7.1 Types of Memory in Agents**

| Memory Type          | Use-case                                 |
| -------------------- | ---------------------------------------- |
| `ConversationBuffer` | Short-term chat history                  |
| `SummaryBuffer`      | Summarized memory for long convos        |
| `VectorStoreMemory`  | Long-term memory using vector embeddings |

**Example:**

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
chat_chain = ConversationChain(llm=llm, memory=memory)

print(chat_chain.predict(input="Hi, I'm John"))
print(chat_chain.predict(input="What's my name?"))  # Remembers!
```

---

### ✅ **7.2 Function Calling / Tool Use (OpenAI Function Support)**

```python
functions = [
  {
    "name": "get_weather",
    "description": "Get weather info",
    "parameters": {
      "type": "object",
      "properties": {"location": {"type": "string"}}
    }
  }
]

response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=[{"role": "user", "content": "Weather in Delhi?"}],
    functions=functions,
    function_call="auto"
)
```

---

### ✅ **7.3 LangChain + Tool Use Example**

```python
from langchain.agents import initialize_agent, load_tools

tools = load_tools(["llm-math", "serpapi"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("What's 15 * 12 and who is the CEO of Microsoft?")
```

---

### ✅ **7.4 LangGraph for Agent State Machines**

Build a stateful workflow with defined transitions:

```python
# Define states: Plan → Research → Summarize → Exit
# Transitions happen based on conditions or timeouts
```

> Best for long-running agents, RPA, decision trees.

Perfect! Let’s wrap it all together and **deploy your AI Agent** like a production-grade app 🚀

---

## 🚀 **8. Deploying AI Agents (FastAPI, Docker, Automation)**

---

### ✅ **8.1 FastAPI + LangChain Agent Backend**

```bash
pip install fastapi uvicorn langchain openai
```

```python
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

app = FastAPI()
llm = ChatOpenAI(model="gpt-3.5-turbo")
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)

class Message(BaseModel):
    input: str

@app.post("/chat")
def chat(msg: Message):
    reply = chain.predict(input=msg.input)
    return {"response": reply}
```

Run server:

```bash
uvicorn main:app --reload
```

---

### ✅ **8.2 Dockerize Your AI Agent**

**Dockerfile**

```Dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t ai-agent .
docker run -p 8000:8000 ai-agent
```

---

### ✅ **8.3 Async Task Handling (Celery + Redis)**

```bash
pip install celery redis
```

**For background agent tasks** like RAG, planning, batch processing

---

### ✅ **8.4 CI/CD Tips**

* Use **GitHub Actions** for automatic deploy on push
* Use **Render, Railway, or AWS ECS/Fargate**
* Secure with `.env`, **rate limits**, **logging**

Here's how to **build a Resume Analyzer AI Agent** step by step:

---

## 🧑‍💼 **Project 5: Resume Analyzer Agent**

### 🎯 Goal:

* Upload resume (PDF or text)
* Extract and analyze content
* Suggest improvements
* Match with real job descriptions

---

### ✅ **Tech Stack**

* `LangChain`, `OpenAI`, `FAISS`, `PyPDF`, `FastAPI`, `Chroma` (optional)
* Frontend (optional): React, Streamlit, or plain HTML upload

---

### 🧱 **Project Structure**

```
resume-analyzer/
├── app/
│   ├── main.py            # FastAPI app
│   ├── analyzer.py        # LangChain logic
│   ├── resume_loader.py   # Parse PDF
│   ├── job_data/          # Sample job descriptions
│   └── templates/         # Frontend (optional)
├── requirements.txt
└── Dockerfile
```

---

### 🔍 **Key Features**

#### 1. **Upload Resume & Parse**

```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("uploads/resume.pdf")
resume_docs = loader.load()
```

#### 2. **Prompt Template for Suggestions**

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "You are a resume expert. Review this resume:\n{resume}\n\nGive top 5 improvement suggestions."
)
```

#### 3. **Job Matching via Embeddings**

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Assume job_descriptions is a list of LangChain documents
embedding = OpenAIEmbeddings()
store = FAISS.from_documents(job_descriptions, embedding)

# Query: match this resume
docs = store.similarity_search("Senior Python Developer, APIs, FastAPI")
```

#### 4. **FastAPI Endpoint**

```python
@app.post("/analyze")
def analyze_resume(file: UploadFile = File(...)):
    # Parse, analyze, return suggestions and job matches
```

---

### 🧪 Output Sample

```json
{
  "improvements": [
    "Add more measurable impact (e.g., 'improved performance by 20%')",
    "Include recent technologies like LangChain, RAG",
    ...
  ],
  "best_matches": [
    {"title": "Python Backend Engineer", "match_score": 0.89},
    {"title": "LLM Developer", "match_score": 0.83}
  ]
}
```






