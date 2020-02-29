FROM python:alpine3.6

RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev
RUN apk add --update --no-cache py3-numpy
RUN apk add --update --no-cache py3-scipy
ENV PYTHONPATH=/usr/lib/python3.6/site-packages

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src src/

RUN python src/backend_api.py

EXPOSE 5000

CMD ["python", "src/backend_api.py"]
