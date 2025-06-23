# SciSynth: Autonomous AI Research Assistant

[![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/<your-username>/<your-repo>/branch/main/graph/badge.svg)](https://codecov.io/gh/<your-username>/<your-repo>)

## 🎯 Project Purpose
A full-stack, production-ready AI system for scientific literature analysis, synthesis, and discovery. Combines LLM-based RAG, symbolic logic, evolutionary algorithms, and multi-agent planning for advanced research workflows.

## 🧠 AI Capabilities & Architecture
- **LLM & RAG**: Retrieval-Augmented Generation with local LLMs (T5, Sentence Transformers)
- **Symbolic & Neuro-Symbolic AI**: Prolog rules, logic consistency, citation networks
- **Reinforcement Learning**: DQN-based paper recommender
- **Evolutionary Algorithms**: Genetic hypothesis generation (DEAP)
- **Agentic AI**: Multi-agent orchestration and cognitive planning
- **ML/DL**: scikit-learn, PyTorch, Transformers

## ⚙️ How to Run
### Local (Python 3.10+)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Docker
```bash
docker build -t scisynth-ai .
docker run -p 8501:8501 scisynth-ai
```

## 🚀 Live Demo
- [Render Cloud Demo](https://your-app.onrender.com)  
- [Hugging Face Spaces](https://huggingface.co/spaces/your-username/scisynth-ai)  
- [Streamlit Cloud](https://share.streamlit.io/your-username/scisynth-ai/main/streamlit_app.py)

## 📦 Tech Stack
- Python 3.10, Streamlit, PyTorch, scikit-learn, HuggingFace Transformers, SentenceTransformers, ChromaDB, DEAP, NetworkX, Prolog, Docker

## ✅ Test Coverage
- All core modules covered by `tests/` (run `pytest`)
- To generate a coverage report: `pytest --cov=app --cov-report=xml`
- To upload to Codecov: `bash <(curl -s https://codecov.io/bash)`

## 🔁 CI/CD
- GitHub Actions for auto-testing and Docker build checks

## 🛠️ Deployment Instructions
### Render Cloud
1. Connect your GitHub repo to [Render](https://render.com/)
2. Select "Web Service", use Dockerfile, and set port to 8501
3. Add a health check: `/ _stcore/health`

### Hugging Face Spaces
1. Create a new Space (Streamlit or Docker)
2. Push your code and requirements.txt (or Dockerfile)
3. App auto-deploys and gives you a public link

### Streamlit Community Cloud
1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your repo and select `streamlit_app.py`
3. App auto-deploys

## 📝 Example Resume/Portfolio Summary
> Autonomous AI Research Assistant: Built a full-stack AI system combining LLM-based RAG, symbolic logic, evolutionary algorithms, and multi-agent planning. Designed and deployed with Docker and CI/CD on Render Cloud. Frontend (Streamlit), backend orchestration (Python), and modular agentic architecture. [Live demo](https://your-app.onrender.com) | [GitHub](https://github.com/your-username/scisynth-ai) | [Docker Hub](https://hub.docker.com/r/your-username/scisynth-ai)

---
*SciSynth: Helping researchers analyze and synthesize scientific literature with advanced AI.* 