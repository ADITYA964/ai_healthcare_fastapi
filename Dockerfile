FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update \
    && apt-get install -y git python3-pip python3.9
    
WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8005

COPY . /app

CMD ["uvicorn", "api:app","--reload"]