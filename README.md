# 🛍️ Simple AI Sales Chatbot (LLaMA 3 + LangChain + FastAPI)

This project is a **lightweight AI-powered sales chatbot** built using **LLaMA 3**, orchestrated through **LangChain**, and served via **FastAPI** with a basic HTML frontend.

It is designed to test the **integration of local LLMs with web frameworks** to simulate a natural, helpful assistant for handling customer sales inquiries.

---

## 🧠 Tech Stack

| Layer       | Technology Used        |
|-------------|------------------------|
| LLM Engine  | LLaMA 3 (local or API) |
| Orchestration | LangChain             |
| API Server  | FastAPI                |
| Frontend    | HTML/CSS + JavaScript  |
| Model Loader | `transformers`, `llama-cpp`, or other compatible runners |

---

## 💡 Features

- 🗣️ LLaMA 3 powered chatbot for handling sales conversations
- ⚙️ LangChain integration for prompt routing and memory
- 🚀 FastAPI backend to serve the chat logic as an API
- 🧪 Simple frontend HTML/JS interface for testing
- 🔄 Stateless or memory-enabled conversations (configurable)

---


## 🚀 Getting Started

1. Clone the Repository

         git clone https://github.com/yourusername/simple-sales-chatbot.git
         cd simple-sales-chatbot

2. Setup Environment

          python -m venv env
          source env/bin/activate
          pip install -r requirements.txt

  
3. Add Environment Variables

          cp .env.example .env

4. Run the Server

          uvicorn src.main:app --reload

Now open your browser at http://localhost:8000

## 💬 Sample Conversation


    User: I'm looking for an affordable smartphone.
    Bot: Sure! What's your budget range?
    User: Around $300.
    Bot: Got it. Here are some models that fit your needs...

## 🧠 How It Works

  1. User types in the web interface.
  2. Message is sent to FastAPI (/chat endpoint).
  3. LangChain processes the prompt using the LLaMA 3 model.
  4. Response is returned and displayed on the page.


## ⚙️ Environment Variables (.env)

    MODEL_PATH=/path/to/llama3/model
    USE_OPENAI=False
    OPENAI_API_KEY=your_key_if_applicable


## 📦 Dependencies

    fastapi
    uvicorn
    langchain
    transformers
    jinja2
    python-dotenv
    llama-cpp-python (optional for local LLaMA)


## 🔧 Customization
  
  - To switch models, update the MODEL_PATH in .env.
  - Modify prompts and behavior in chatbot_engine.py.
  - Add memory or retrieval with LangChain tools if desired.

## 🛡️ License


MIT License – Feel free to use, fork, and improve this for your own AI assistant projects.

## 🤝 Contributing

Pull requests and issues are welcome! Help improve lightweight LLaMA chatbot deployment.



