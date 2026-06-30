# 🤖 Chatbutter AI

A stateful, tool-using AI chatbot built with **LangGraph**, **Streamlit**, and **Groq/Gemini**, featuring persistent multi-thread conversations, PDF-based RAG, real-time stock/web data, and human-in-the-loop approval for sensitive actions.

🔗 **Live demo:** [http://ec2-54-165-101-41.compute-1.amazonaws.com:8501](http://ec2-54-165-101-41.compute-1.amazonaws.com:8501)

---

## ✨ Features

- **Stateful conversations** — chat history is persisted per-thread using a SQLite-backed LangGraph checkpointer, so conversations survive restarts.
- **Multi-thread chat sessions** — switch between independent conversations from the sidebar, just like ChatGPT's chat history.
- **Tool-calling agent** powered by Groq/Gemini, with access to:
  - 🧮 **Calculator** — evaluates math expressions.
  - 📈 **Stock price lookup** — real-time quotes via Alpha Vantage.
  - 🌐 **Web search** — live web results via Tavily.
  - 📄 **PDF RAG** — upload a PDF and ask questions grounded in its content (FAISS + HuggingFace embeddings).
  - 💸 **Stock purchase (simulated)** — demonstrates **human-in-the-loop approval**: the agent pauses and asks for explicit user confirmation before "executing" a trade.
- **Human-in-the-loop interrupts** — sensitive tool calls pause the graph and surface an approve/cancel UI before continuing.
- **Per-thread PDF knowledge bases** — each chat thread can have its own uploaded document.

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌────────────────┐
│  Streamlit  │◄────►│  LangGraph Agent  │◄────►│  Groq / Gemini  │
│   (app.py)  │      │   (backend.py)    │      │      LLM        │
└─────────────┘      └──────────────────┘      └────────────────┘
                              │
                  ┌───────────┼───────────┐
                  ▼           ▼           ▼
            ┌─────────┐ ┌──────────┐ ┌──────────┐
            │ Tavily  │ │  FAISS   │ │  Alpha   │
            │  Search │ │ PDF RAG  │ │ Vantage  │
            └─────────┘ └──────────┘ └──────────┘
```

State and chat history are persisted via `SqliteSaver`, so each thread's conversation graph can be resumed at any time — including mid-interrupt (e.g. a pending stock purchase approval).

---

## 🧰 Tech Stack

| Layer            | Technology                                 |
| ---------------- | ------------------------------------------ |
| UI               | Streamlit                                  |
| Orchestration    | LangGraph                                  |
| LLM              | Groq (Llama) / Google Gemini               |
| Vector store     | FAISS                                      |
| Embeddings       | HuggingFace (`all-MiniLM-L6-v2`)           |
| Tools            | Tavily Search, Alpha Vantage, custom tools |
| Persistence      | SQLite (LangGraph checkpointer)            |
| Observability    | LangSmith                                  |
| Containerization | Docker                                     |
| CI/CD            | GitHub Actions (self-hosted runner on EC2) |
| Hosting          | AWS EC2                                    |
| Image registry   | Docker Hub                                 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized run)
- API keys for: Groq, Google Gemini, Tavily, Alpha Vantage, LangSmith (optional but recommended)

### 1. Clone the repository

```bash
git clone https://github.com/ahmedbaig2004/chatbutter-ai.git
cd chatbutter-ai
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your-groq-api-key
GOOGLE_API_KEY=your-google-api-key
TAVILY_API_KEY=your-tavily-api-key
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-api-key
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ChatButter
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

> ⚠️ Never commit your `.env` file. It's already excluded via `.gitignore`.

### 3. Run locally (without Docker)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### 4. Run with Docker

```bash
docker build -t chatbutter-ai .
docker run -d -p 8501:8501 --env-file .env --name chatbutter-app chatbutter-ai
```

---

## 🔄 CI/CD Pipeline

This project is automatically tested, built, and deployed via **GitHub Actions**:

1. **Lint & Test** — runs `flake8` and `pytest` on every push to `main`.
2. **Build & Push** — builds a Docker image and pushes it to Docker Hub, tagged with both `latest` and the commit SHA.
3. **Deploy** — runs on a **self-hosted GitHub Actions runner installed on the EC2 instance**, which pulls the new image and redeploys the container with a **health check and automatic rollback** if the new version fails to come up healthy.

See [`.github/workflows/cicd.yaml`](.github/workflows/cicd.yaml) for the full pipeline definition.

---

## 📂 Project Structure

```
.
├── app.py                  # Streamlit frontend
├── backend.py               # LangGraph agent, tools, and graph definition
├── requirements.txt
├── Dockerfile
├── .github/
│   └── workflows/
│       └── cicd.yaml        # CI/CD pipeline
└── README.md
```

---

## 🛣️ Roadmap

- [ ] Add automated test coverage for tools and graph nodes
- [ ] Support multiple file types beyond PDF in RAG
- [ ] Add user authentication for multi-user deployments
- [ ] Serve behind HTTPS via a reverse proxy (nginx/Caddy) + domain name

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Streamlit](https://streamlit.io/)
- [Groq](https://groq.com/)
- [Tavily](https://tavily.com/)
