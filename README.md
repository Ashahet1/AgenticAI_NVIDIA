# 🏋️‍♂️ Workout InjuryIntel — AI-Powered Injury Analysis & Action Plan Generator

> **Smart multi-agent system** that analyzes workout-related pain or discomfort, diagnoses likely form or injury issues, and provides a personalized action plan — all powered by reasoning agents.

---
[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-Click%20Here-blueviolet?style=for-the-badge)](https://rebeca-groutiest-incorporeally.ngrok-free.dev/)
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BhrEQneXIRvGKZxGQxZ4dNuq2j8WfLJC#scrollTo=uqm4g9hOZTgw)

## Motivation

Athletes and fitness enthusiasts often experience pain or discomfort during exercises and struggle to identify the cause. For instance:

> "My knee hurts after squats."  
> "My shoulder clicks during overhead press."  
> "Should I continue training or rest?"

Most users rely on unverified online sources, leading to confusion and inconsistent advice. **Workout InjuryIntel** addresses this gap using an **AI-driven, multi-agent system** that emulates clinical reasoning and biomechanics expertise.

The goal is to deliver **real-time, evidence-based recommendations** that combine natural language understanding, movement analysis, and medical knowledge retrieval — helping users train safely while minimizing injury risk.

---

## System Overview

Workout InjuryIntel operates as a **multi-agent reasoning system** designed to simulate how a sports medicine expert evaluates exercise-related pain.  
The system integrates several specialized agents that collaborate under a central orchestration module called the **Master Orchestrator**.

Each agent performs a specific task in a structured sequence that mirrors professional diagnostic reasoning — from symptom parsing to prescription planning.

### Multi-Agent Pipeline

| Step | Agent | Description |
|------|--------|-------------|
| 1 | **ParsingAgent** | Extracts exercise type, pain location, side, and timing from natural-language input. |
| 2 | **FormAnalysisAgent** | Analyzes biomechanical movements to identify likely form deviations or muscle imbalances. |
| 3 | **InjuryDiagnosisAgent** | Uses contextual reasoning to hypothesize the most probable injury and estimate confidence level. |
| 4 | **ResearchAgent** | Queries the knowledge base and scientific literature using retrieval embeddings to validate findings. |
| 5 | **PrescriptionAgent** | Generates a customized recovery plan with progressive exercises, red flags, and referral recommendations. |

### Master Orchestrator

The **Master Orchestrator** serves as the control layer.  
It manages the flow of data between agents, interprets LLM outputs, and determines execution order based on system confidence and user input completeness.

This orchestration includes:
- A **PlannerAgent**, which dynamically decides which module to execute next.
- A **ConversationManager**, which gathers missing context by asking relevant follow-up questions.
- A **ReasoningController**, which enforces a “reason → reflect → retry” loop for consistent inference quality.

---

## Features and Capabilities

Workout InjuryIntel integrates advanced reasoning and retrieval mechanisms to provide structured, medically aligned insights for workout-related pain and injury analysis.  
Its core features emphasize modularity, explainability, and real-world applicability.

### Key Features

**1. Multi-Agent Orchestration**  
Implements a sequence of specialized agents managed by the `MasterOrchestrator`. Each agent performs a unique reasoning step, enabling human-like contextual flow between analysis stages.

**2. Conversational Understanding**  
Uses a dynamic `ConversationManager` that identifies missing information and automatically asks follow-up questions in natural language, ensuring minimal user input fatigue.

**3. Planner-Driven Reasoning Control**  
A `PlannerAgent` governs workflow by evaluating each agent’s output and confidence level, skipping redundant modules when sufficient context exists. This minimizes unnecessary LLM calls and improves efficiency.

**4. Integrated Retrieval and Knowledge Tools**  
Combines a local `KnowledgeBaseTool` (pre-indexed PDFs and biomechanical reports) with a `WebSearchTool` that performs controlled, context-aware external lookups to supplement reasoning.

**5. Structured Evidence Generation**  
The `ResearchAgent` validates inferences through retrieval embeddings, referencing relevant literature or clinical studies. This enforces traceable, evidence-based recommendations.

**6. Personalized Action Planning**  
The `PrescriptionAgent` creates a tailored 7-day training and recovery plan based on the diagnosed condition, severity indicators, and biomechanical causes.

**7. Modular Extensibility**  
Each agent is independently replaceable and can be upgraded with different models, retrievers, or medical datasets without disrupting the orchestration logic.

**8. Web UI Integration**  
A lightweight, interactive frontend (HTML + JavaScript) connects directly to the Flask backend, displaying stepwise reasoning updates and recommendations in real time.

---

## Technical Architecture and Stack

Workout InjuryIntel is built on a modular.
The platform combines containerized AI agents, reasoning orchestration, and retrieval-based knowledge systems.


### Deployment and Runtime

| Component | Platform / Technology | Purpose |
|------------|-----------------------|----------|
| **Model Runtime** | NVIDIA NIM (Llama-3.1-Nemotron-Nano-8B-v1) | High-performance reasoning engine hosted in an isolated inference container. |
| **Container Management** | AWS Elastic Kubernetes Service (EKS) | Deploys multi-agent microservices and manages scaling, health checks, and rolling updates. |
| **Backend Service** | Flask (Python) | Exposes REST endpoints (`/chat`, `/health`) for the Web UI and handles request routing. |
| **Frontend** | HTML + JavaScript | Provides an interactive user interface for input, progress display, and AI recommendations. |
| **Retrieval Mechanism** | NVIDIA Embedding API (`nv-embedqa-e5-v5`) | Enables vector search across biomechanical PDFs and research materials. |
| **Knowledge Index** | FAISS-based local index + ResearchAgent | Manages document embeddings for fast evidence retrieval. |
| **Storage and Logs** | AWS S3 + CloudWatch | Persists artifacts, model responses, and structured action plans. |
| **Networking** | API Gateway + HTTPS ingress | Routes web traffic securely to the orchestrator and agents. |


---

## 🧩 Folder Structure

```plaintext
WorkoutFormChecker/
│
├── MasterOrchestra/
│   ├── base_agent.py              # Base class for all agents with shared LLM call interface
│   ├── conversation_manager.py    # Manages dialogue context and clarification questions
│   ├── planner_agent.py           # Decides next agent to execute based on reasoning flow
│   ├── parsing_agent.py           # Extracts exercise, pain details, and timing
│   ├── form_analysis_agent.py     # Evaluates movement form and biomechanical stress points
│   ├── injury_diagnosis_agent.py  # Infers probable injury and confidence score
│   ├── research_agent.py          # Retrieves relevant biomechanical or clinical studies
│   ├── prescription_agent.py      # Generates actionable weekly recovery and prevention plan
│   ├── master.py                  # Core orchestrator that coordinates agent interactions
│   ├── server.py                  # Flask-based backend exposing REST endpoints
│   ├── tools/                     # Utility scripts (e.g., KnowledgeBaseTool, WebSearchTool)
│   ├── knowledge_base/            # Local indexed PDFs and embeddings for retrieval
│   ├── logs/                      # Optional logging directory for request tracking
│   └── requirements.txt           # Python dependencies
│
├── frontend.html                  # Lightweight HTML+JS UI for user interaction
├── Images/                        # Architecture diagrams and UI screenshots
└── README.md                      # Project documentation and setup guide
```
---
## Architecture Diagram
![AWS SageMaker Architecture](https://github.com/Ashahet1/AgenticAI_NVIDIA/blob/main/WorkoutFormChecker/Images/ArchitectureDesign.png)
---

## ⚙️ Installation & Setup

### 1️⃣ Clone or copy the repo
```bash
git clone https://github.com/Ashahet1/AgenticAI_NVIDIA/WorkoutFormChecker.git
cd WorkoutFormChecker/MasterOrchestra
```


