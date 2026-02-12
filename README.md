# 🤖 AZ-900 Guru: Agentic RAG Chatbot

An intelligent Retrieval-Augmented Generation (RAG) system designed to help students master the **Azure AZ-900 Fundamentals** exam. This project features a unique "Self-Refactoring" pipeline where a team of AI agents converts an experimental Jupyter Notebook into a production-ready application.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM Orchestration** | [LangChain](https://www.langchain.com/) |
| **Inference Engines** | [Cerebras](https://cerebras.ai/) (GPT-OSS-120B) & [Mistral](https://mistral.ai/) |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) |
| **UI Framework** | [Gradio](https://gradio.app/) (Glass Theme) |
| **Agentic Framework** | [CrewAI](https://crewai.com/) |

---

## 🧠 Project Architecture

The project follows a two-stage evolution:

### 1. The Prototype (Jupyter Notebook)
The core RAG logic was developed in 25 cells, focusing on:
* **Ingestion**: Loading `Azure.txt` via `TextLoader`.
* **Processing**: `RecursiveCharacterTextSplitter` (2000 chunk size, 200 overlap).
* **Storage**: Persistent `Chroma` collection named `azure_prep`.
* **Persona**: A specialized "Azure Teacher" system prompt for pedagogical accuracy.

### 2. The Refactoring Crew (CrewAI)
A specialized multi-agent team was configured to productionize the code:

* **Senior LLM Engineer (Interpreter)**: Scans the notebook cells to create a blueprint for translation, identifying dependencies and core logic.
* **Backend Developer (Python Expert)**: Refactors the logic with a focus on **Clean Code**, optimizing the retrieval functions and variable management.
* **Gradio Specialist (Frontend)**: Integrates the backend into a sleek UI, ensuring high-performance streaming and visual design.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* [Cerebras API Key](https://cloud.cerebras.ai/)
* [Mistral API Key](https://console.mistral.ai/)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Enyo2004/AZ-900-Guru-Agentic-RAG-Chatbot.git
   
