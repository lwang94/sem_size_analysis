FROM python:alpine3.6

RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev

COPY requirements.txt .

RUN pip install --upgrade pip
RUN apk update && apk add --update py-pip && apk add --no-cache gcc musl-dev make && ln -s /usr/include/locale.h /usr/include/xlocale.h \
&& pip install -r requirements.txt && apk del gcc musl-dev make

COPY src src/

RUN python src/backend_api.py

EXPOSE 5000

CMD ["python", "src/backend_api.py"]
