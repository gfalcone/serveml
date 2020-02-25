FROM python:3.7-buster

COPY requirements.txt /app/

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0"]