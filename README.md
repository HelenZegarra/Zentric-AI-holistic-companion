# Zentric-AI-holistic-companion
Zentric AI Holistic bot  
**[Live Demo](https://zentricwordpressembed-production.up.railway.app/) · 
[Case Study](https://helenzegarra.com/portfolio/zentric-ai-health-agent/)**

### IMPORTANT WORK IN PROGRESS
## Made to run locally using LM Studio 

# Zentric
A holistic AI companion for chronic illness management, blending LangGraph orchestration with empathetic UX to transform healthcare support into a warm, human-centered experience.

---
## The problem

Managing a chronic illness means navigating an overwhelming amount of health information — medication schedules, dietary restrictions, symptom  tracking, nutrition guidance — while already dealing with the emotional weight of being unwell. Most AI health tools respond like search engines: fast, generic, and emotionally tone-deaf.

Zentric is built around a different premise: that the architecture of an AI system determines whether it feels safe to talk to. A non-deterministic LLM with no guardrails will hallucinate dosage information, give contradictory advice across sessions, and respond to vulnerable users without appropriate emotional calibration. That is not a UX problem — it is an engineering problem.

Zentric solves it with deterministic state-machine architecture over LangGraph, a curated RAG knowledge base grounded in clinical sources, and a conversational UX designed for trust.


## 🚀 Features

*   **Intelligent Orchestration:** Advanced state management handling robust human-in-the-loop AI interactions.
*   **Modular Architecture:** Clean segregation between decoupled front-end experiences and scalable back-end services.
*   **Context-Aware Processing:** Designed to leverage dynamic data streaming and efficient API routing for real-time responsiveness.

---

## 🛠️ Tech Stack

*   **Front-End:** [React / Next.js / HTML5]
*   **Back-End:** [ Node.js / Express / Python/ Langraph/ LM Studio]
*   **AI Integration:** Google Gemini API
*   **State & Database:** [American Diabetes Association (https://professional.diabetes.org/)/ Spoontacular API]

---

## Architecture
User input
↓
React.js frontend
↓
FastAPI / Python backend
↓
LangGraph state machine
(deterministic conversation flow — controls what
the LLM can and cannot do at each state)
↓
├── RAG pipeline
│   ├── Query embedding
│   ├── ChromaDB vector retrieval
│   └── Curated knowledge base
│       ├── American Diabetes Association docs
│       └── Spoonacular nutrition API
↓
Google Gemini API
(LLM response generation — bounded by
retrieved context and state constraints)
↓
Emotionally calibrated response → user

## Key design decisions

**Deterministic over probabilistic** — LangGraph state machines bound LLM behavior at each conversation node. The system cannot randomly drift into an inappropriate response path.

**Retrieval-grounded responses** — all health and nutrition guidance is retrieved from curated clinical sources (American Diabetes Association) and real-time nutrition data (Spoonacular API) and other ressources to be tested before being passed to the LLM as context. The model explains, it does not invent.

**Emotional calibration** — conversation states include explicit emotional context handling. A user expressing distress routes differently than a user asking a factual question, an agent should manage positevely both types of responses to correlate to each user state.


## 📦 Installation & Setup

Follow these steps to get a local development copy running on your machine.

### Prerequisites

Ensure you have the following installed:
*   [Node.js](https://nodejs.org/) (v18+ recommended)
*   Git

### Live Gradio preview test Work in Progress: 
* Live Gradio ( https://zentricwordpressembed-production.up.railway.app/)


### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR-USERNAME/zentric.git](https://github.com/YOUR-USERNAME/zentric.git)
cd zentric
