# 🏋️‍♂️ Workout Form Checker — AI-Powered Injury Analysis & Action Plan Generator

> **Smart multi-agent system** that analyzes workout-related pain or discomfort, diagnoses likely form or injury issues, and provides a personalized action plan — all powered by reasoning agents.

---
[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-Click%20Here-blueviolet?style=for-the-badge)](https://rebeca-groutiest-incorporeally.ngrok-free.dev/)
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
- Light/Dark feature UI
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
│   ├── server.py                # Flask backend
│   ├── master.py                # Master orchestrator managing all agents
│   ├── base_agent.py            # Shared logging + utility class
│   ├── parsing_agent.py         # Step 1 - Parse user input
│   ├── form_analysis_agent.py   # Step 2 - Analyze form
│   ├── injury_diagnosis_agent.py# Step 3 - Diagnose issue
│   ├── research_agent.py        # Step 4 - Gather evidence
│   ├── prescription_agent.py    # Step 5 - Generate plan
│
├── frontend.html                # Simple web interface with progress animation
└── README.md                    # You are here 😄
```


---

## ⚙️ Installation & Setup

### 1️⃣ Clone or copy the repo
```bash
git clone https://github.com/Ashahet1/AgenticAI_NVIDIA/WorkoutFormChecker.git
cd WorkoutFormChecker/MasterOrchestra
```


