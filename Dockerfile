FROM python:alpine3.6

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src src/

RUN python src/backend_api.py

EXPOSE 5000

CMD ["python", "src/backend_api.py"]
