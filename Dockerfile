FROM python:3.6-slim-stretch

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY src src/

RUN python -m src.backend_api

EXPOSE 5000

CMD ["python", "-m", "src.backend_api"]