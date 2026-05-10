FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN ["pip", "install",  "--no-cache-dir", "-r", "requirements.txt"]

COPY data/ data/
COPY src/ src/

ENV EXECUTION_CMD=""

ENTRYPOINT ["sh", "-c", " $EXECUTION_CMD"]