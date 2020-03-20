FROM python:3.7-buster

COPY requirements.txt /tmp/
COPY requirements-test.txt /tmp/

RUN apt-get update && apt-get install sqlite3 && pip install -r /tmp/requirements.txt

# for testing
RUN pip install -r /tmp/requirements-test.txt

COPY . /app/

WORKDIR /app

CMD ["bash", "create_dev_environment.sh"]