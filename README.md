# Quizzy

Quizzy is an AI-powered virtual interviewer built to simulate personalized interviews through a robust integration of Retrieval-Augmented Generation (RAG), multi-modal analysis (vision, audio, text), and intelligent evaluation systems. It processes resumes and job descriptions, evaluates candidates in real-time, and generates detailed feedback, scoring, and job suggestions.Quizzy delivers an end-to-end mock interview experience that is hyper-personalized and interactive.

Built on Django, Quizzy orchestrates a modular architecture where core machine learning components are managed in a dedicated ML repository. These components are dynamically pulled and tracked using MLflow, DagsHub, and MLOps pipelines, ensuring reproducibility, scalability, and version control for both data and models.

---

## Demo

**Video Overview**  
![Quizzy Demo](https://github.com/RijoSLal/my-portfolio/blob/main/images/quizzy.png)
Watch the full walkthrough of how Quizzy functions and performs an AI-driven interview:  
[YouTube Demo Link][(https://youtu.be/JGKYGQfVkoU)](https://youtu.be/JGKYGQfVkoU)

---

## Architecture and Features

### Workflow

1. **Input Collection**: Users provide their resume, job description, and preferred interview time.
2. **Resume Analysis**:
   - Name and domain extraction
   - Resume-job description similarity using MXBAI embeddings (cosine similarity)
3. **Interview Phase**:
   - Dynamic question generation via Groq's LLaMA 70B model
   - Document retrieval from ChromaDB with Gemini embeddings
   - Real-time TTS (Edge TTS) and STT (Whisper)
   - Vision-based posture and emotion detection using MediaPipe and MobileNet
   - Profile summarization using HuggingFace model
4. **Post Interview**:
   - Score computation and evaluation report
   - Suggestions for improvement
   - Curated job recommendations from LinkedIn scraping based on candidate profile

### ML and MLOps Stack

- **Model Management**: MLflow
- **Experiment Tracking & Versioning**: DagsHub
- **Vision Models**: MobileNet (transfer learned), MediaPipe
- **Text Models**: Groq (LLaMA 70B), Gemini Embeddings
- **TTS/STT**: Edge TTS and Whisper
- **Resume Matching**: MXBAI embeddings
- **Document Store**: ChromaDB

> ML code and training pipelines are managed in a separate repository:  
> [https://github.com/RijoSLal/Quizzy_MLOPS](https://github.com/RijoSLal/Quizzy_MLOPS)  
> Models and data are versioned and stored at:  
> [https://dagshub.com/slalrijo2005/Quizzy_MLOPS](https://dagshub.com/slalrijo2005/Quizzy_MLOPS)

---

### ⚠️ Minimum System Requirements

- **CPU:** 12 threads (e.g., 4P + 4E cores, 2.0 GHz+)
- **RAM:** 6 GB
- **GPU:** Integrated, 48 EUs+, OpenGL 4.5+

> ⚠️ Systems below these specs may face performance issues or instability.

---

## Deployment

To configure a **production-ready self-hosted environment using NGINX and Cloudflare Tunnel**, refer to my detailed blog post:  
[Self-hosting Django with NGINX & Cloudflare Tunnel](https://rijo.hashnode.dev/self-hosting-django-with-nginx-cloudflare-tunnel-configure-a-production-ready-server)

---

## Local Development

### Requirements

- Python 3.8+
- pip
- virtualenv

### Setup

```bash
git clone https://github.com/RijoSLal/quizzy.git
cd quizzy
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Set Environment Variables

Create a `.env` file:

```env
APIKEY=gemini_api_key
GROQ=groq_api_key
GROQQ=groq_api_key
GROQ_API_KEY=groq_api_key
DJANGO=django_secret_key
```

### Run Tests

```bash
python manage.py test
```

---

### Run the Server

```bash
python manage.py runserver
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t quizzy .
```

### Run with Environment File

```bash
docker run --env-file .env quizzy
```

### Run Tests

```bash
docker run --env-file .env quizzy python manage.py test
```

---

## Directory Structure

```
quizzy/
├── chroma_langchain_db/
├── interview/
|   ├── urls.py
|   ├── no_stream_camera_capture.py
│   ├── camera_capture.py
│   ├── resume_management.py
│   ├── retriever.py
│   ├── scrape.py
│   ├── speech.py
│   ├── vectordb.py
│   └── views.py
├── manage.py
├── quizzy/
│   ├── settings.py
│   └── urls.py
├── static/
|   ├── images/
│   └── styling/
├── templates/
│   ├── home.html
│   ├── interview.html
│   ├── resume.html
│   └── score.html
├── Dockerfile
├── .dockerignore
├── .env
├── requirements.txt
└── preview.png         # Optional preview image for README
```

---

## Contribution

Contributions are welcome. Please open an issue or submit a pull request if you would like to suggest improvements or contribute features. Ensure your contributions follow the project’s code standards and include relevant documentation.

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.

---
