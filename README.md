# SciSynth: Autonomous AI Research Assistant

[![CI](https://github.com/Avinash291020/SciSynth-AI-Research-Assistant/actions/workflows/docker-push.yml/badge.svg)](https://github.com/Avinash291020/SciSynth-AI-Research-Assistant/actions/workflows/docker-push.yml)
[![Documentation](https://github.com/Avinash291020/SciSynth-AI-Research-Assistant/actions/workflows/docs.yml/badge.svg)](https://github.com/Avinash291020/SciSynth-AI-Research-Assistant/actions/workflows/docs.yml)

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

### Docker Hub (Latest Release)
```bash
docker run -p 8501:8501 avinash76543/scisynth-ai:latest
```

## 🚀 Live Demo
- **Docker Hub**: `avinash76543/scisynth-ai:latest`
- **GitHub Actions**: Automated CI/CD with Docker deployment
- **Documentation**: Auto-generated API docs via pdoc

## 📦 Tech Stack
- Python 3.10, Streamlit, PyTorch, scikit-learn, HuggingFace Transformers, SentenceTransformers, ChromaDB, DEAP, NetworkX, Prolog, Docker

## ✅ Test Coverage
- All core modules covered by `tests/` (run `pytest`)
- To generate a coverage report: `pytest --cov=app --cov-report=xml`
- To upload to Codecov: `bash <(curl -s https://codecov.io/bash)`

## 🔁 CI/CD
- **GitHub Actions**: Auto-testing, Docker builds, and documentation generation
- **Docker Hub**: Automated image publishing with version tags
- **Documentation**: Auto-generated API documentation

## 🛠️ Deployment Instructions
### Docker Hub (Recommended)
1. Pull the latest image: `docker pull avinash76543/scisynth-ai:latest`
2. Run locally: `docker run -p 8501:8501 avinash76543/scisynth-ai:latest`
3. Access at: http://localhost:8501

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
> Autonomous AI Research Assistant: Built a full-stack AI system combining LLM-based RAG, symbolic logic, evolutionary algorithms, and multi-agent planning. Designed and deployed with Docker and CI/CD on Docker Hub. Frontend (Streamlit), backend orchestration (Python), and modular agentic architecture. [Docker Hub](https://hub.docker.com/r/avinash76543/scisynth-ai) | [GitHub](https://github.com/Avinash291020/SciSynth-AI-Research-Assistant)

---
*SciSynth: Helping researchers analyze and synthesize scientific literature with advanced AI.* 

**Latest Update**: Fixed CI/CD workflows and documentation generation. All systems operational! 🚀 