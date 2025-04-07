
FROM python:3.10

WORKDIR /app 

COPY . .

RUN apt-get update && apt-get install -y curl libgl1-mesa-glx libglib2.0-0

# install the actual ollama CLI tool
RUN curl -fsSL https://ollama.com/install.sh | sh

# add it to PATH
ENV PATH="/root/.ollama/bin:$PATH"

RUN pip install --no-cache-dir -r requirements.txt 

# pull models using the CLI
RUN ollama pull llama3.2:1b && ollama pull mxbai-embed-large:latest

EXPOSE 8000

CMD ["uvicorn", "quizzy.asgi:application", "--host", "0.0.0.0", "--port", "8000"]