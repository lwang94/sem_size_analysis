FROM python:alpine3.6

RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src src/

RUN python src/backend_api.py

EXPOSE 5000

CMD ["python", "src/backend_api.py"]
