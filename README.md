# 🏋️‍♂️ Workout InjuryIntel — AI-Powered Injury Analysis & Action Plan Generator

> **Smart multi-agent system** that analyzes workout-related pain or discomfort, diagnoses likely form or injury issues, and provides a personalized action plan — all powered by reasoning agents.

---
[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-Click%20Here-blueviolet?style=for-the-badge)](https://rebeca-groutiest-incorporeally.ngrok-free.dev/)
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BhrEQneXIRvGKZxGQxZ4dNuq2j8WfLJC#scrollTo=uqm4g9hOZTgw)

## 🚀 Motivation

Every athlete or fitness enthusiast has faced it —  
> “My lower back hurts after deadlifts,”  
> “I feel shoulder pain during bench press,”  
> “Why does my stomach hurt when I do crunches?”

Most people guess the cause. Some Google it. Few get expert guidance.

This project bridges that gap — using **AI agents** that simulate how a sports medicine professional would think:
- Parse your workout and symptoms.
- Analyze likely form errors.
- Diagnose potential issues.
- Back findings with real research.
- Recommend an action plan to recover safely.

The goal is **real-time, evidence-based feedback** that keeps training both **safe** and **effective**.

---

## 🧠 Multi-Agent Architecture

The system follows a structured reasoning pipeline through **five specialized agents**, orchestrated by `MasterOrchestrator`:

| Step | Agent | Role |
|------|--------|------|
| 🧩 1 | **ParsingAgent** | Extracts exercise, pain location, timing, and intensity from natural language. |
| 🏋️ 2 | **FormAnalysisAgent** | Evaluates biomechanical issues (form breakdown, overuse, instability). |
| 🩺 3 | **InjuryDiagnosisAgent** | Uses reasoning to identify the most likely injury and confidence level. |
| 🔬 4 | **ResearchAgent** | Searches the web for scientific or medical support for the diagnosis. |
| 📋 5 | **PrescriptionAgent** | Creates a week-long action plan with do/don’t recommendations and professional referral notes. |

Each agent is modular and can be independently upgraded or replaced for fine-tuning or specialization.

---

## ✨ Features

- 🧠 **Multi-agent orchestration** with structured reasoning flow.
- 🔍 **Dynamic symptom parsing** — understands free-text natural language inputs.
- 🩹 **Evidence-backed diagnosis** using live research queries.
- 💪 **Personalized recovery plan** generation.
- 🌐 **Web UI integration** — interactive frontend with progress animation.
- ⚡ **Flask backend** with `/run` endpoint for analysis orchestration.
- 🔄 **CORS-enabled** for direct connection via ngrok or local frontend.
---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-------------|
| Frontend | HTML, CSS (Inter font, light/dark mode), Vanilla JS |
| Backend | Python (Flask + Flask-CORS) |
| Agents | Custom Python modules orchestrated via `MasterOrchestrator` |
| AI Model | LLM (**llama-3.1-nemotron-nano-8B-v1** in reasoning mode, deployed as an **NVIDIA NIM** microservice with at least one **Retrieval Embedding NIM |
| Hosting | Amazon Sagemaker|

---

## 🧩 Folder Structure

```plaintext
WorkoutFormChecker/
│
├── MasterOrchestra/
│   ├── server.py                   # 🌐 Flask backend exposing /run endpoint
│   ├── master.py                   # 🧠 Agentic Orchestrator with integrated ReasoningController
│   │                                # (Reason → Reflect → Retry loop with confidence feedback)
│   ├── base_agent.py               # 🧩 Base agent class with logging and safe execution
│   ├── parsing_agent.py            # 🧾 Step 1 - Parse workout input and reasoning fields
│   ├── form_analysis_agent.py      # 🏋️ Step 2 - Analyze biomechanics and detect form issues
│   ├── injury_diagnosis_agent.py   # 🩺 Step 3 - Diagnose probable injury and root cause
│   ├── research_agent.py           # 🔍 Step 4 - Retrieve supporting biomechanical/medical evidence
│   ├── prescription_agent.py       # 📋 Step 5 - Generate personalized recovery & prevention plan
│
├── frontend.html                   # 💻 Interactive web UI with animated reasoning progress
├── requirements.txt                # 📦 Flask, Flask-CORS, and dependencies
└── README.md                       # 🧭 Project overview and setup instructions

```
---
## Architecture Diagram
![AWS SageMaker Architecture](https://github.com/Ashahet1/AgenticAI_NVIDIA/blob/main/WorkoutFormChecker/Images/aws_sagemaker_architecture%20(2).svg)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone or copy the repo
```bash
git clone https://github.com/Ashahet1/AgenticAI_NVIDIA/WorkoutFormChecker.git
cd WorkoutFormChecker/MasterOrchestra
```


