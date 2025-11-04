# 🤖 Q&A BOT – CONVERGE

A scalable **RAG-based Q&A system** built with **FastAPI** and **LangChain**, designed for **call center agents** to answer questions about uploaded **contract PDFs**.  
The system extracts metadata, performs smart chunking, embeds data using **OpenAI embeddings**, and retrieves structured answers enriched with **contract details**.

---

## 🚀 Features

- 📄 **Upload PDF contracts** (via FastAPI endpoint)
- ✂️ **Smart Chunking** with `RecursiveCharacterTextSplitter` (semantic + overlapping)
- 🧠 **Embeddings & RAG pipeline** using `LangChain` + `OpenAI`
- 🧮 **FAISS vector store** (local, persistent between sessions)
- 🗂️ **Automatic metadata extraction** (client name, ID, region, dates)
- 🧱 **Structured JSON answers** (client insights + global contract stats)
- ⚙️ **Extensible design** (easily connect live databases or agent tools)
- 🧰 **Modular codebase** (clean architecture, ready for production)

---

## 🏗️ Project Structure

