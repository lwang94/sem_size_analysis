FROM python:alpine3.6

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade -r requirements.txt

COPY app app/

RUN python src/backend_api.py

EXPOSE 5000

CMD ["python", "src/backend_api.py"]
