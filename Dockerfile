FROM python:3.7-buster

COPY requirements.txt /app/

WORKDIR /app

RUN apt-get update && apt-get install -y mariadb-server nano less telnet && pip install -r requirements.txt && mkdir /app/runs

COPY tests/train.py /app
COPY mlserve /app
COPY env.sh /app
COPY bootstrap.sh /app

EXPOSE 5000

ENTRYPOINT ["/app/bootstrap.sh"]

CMD ["mlflow", "server", "--backend-store-uri", "mysql://root:root@mysql/mlflow", "--default-artifact-root", "s3://drivy-data-dev/mlflow/app/runs", "-h", "0.0.0.0"]