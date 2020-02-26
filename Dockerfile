FROM python:3.7-buster

COPY requirements.txt /app/

WORKDIR /app

RUN apt-get update && apt-get install -y mariadb-server && pip install -r requirements.txt && mkdir /app/runs

COPY tests/train.py /app/

EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", "mysql://root:root@mysql/mlflow", "--default-artifact-root", "/app/runs", "-h", "0.0.0.0"]