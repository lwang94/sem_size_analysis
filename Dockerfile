FROM python:alpine3.6


COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY app app/

RUN python src/backend_api.py

EXPOSE 5000

CMD ["python", "src/backend_api.py"]
