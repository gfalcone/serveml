FROM python:3.7-buster

COPY requirements.txt /app/
COPY tests/train.py /app/

WORKDIR /app

RUN apt-get update

RUN pip install -r requirements.txt

RUN python3 /app/train.py

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", "mysql://mlflow:mlflow@mysql:3306/mlflow"]