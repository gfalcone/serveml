FROM python:3.7-buster as base

COPY requirements.txt /tmp/
COPY requirements-test.txt /tmp/

RUN apt-get update && apt-get install sqlite3 && pip install -r /tmp/requirements.txt

# for testing
FROM base as test

RUN pip install -r /tmp/requirements-test.txt

COPY . /app/

WORKDIR /app

RUN bash create_dev_environment.sh && \
    bash run_tests.sh

# final image
FROM base as final

COPY . /app/

WORKDIR /app

CMD ["bash", "bootstrap.sh"]