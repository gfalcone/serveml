FROM python:3.7-buster

COPY requirements.txt /tmp/

RUN apt-get update && apt-get install -y mariadb-server nano less telnet && pip install -r /tmp/requirements.txt

WORKDIR /app

ENV MLFLOW_TRACKING_URI=http://localhost:5000

EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", "mysql://root:root@mysql/mlflow", "--default-artifact-root", "s3://drivy-data-dev/mlflow/app/runs", "-h", "0.0.0.0"]