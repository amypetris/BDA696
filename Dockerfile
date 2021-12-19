FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y mariadb-client

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        python3 \
        python3-pip \
        python3-dev \
        mariadb-client \
    && rm -rf /var/lib/apt/lists/*

COPY final.sql .
COPY baseball.sql .
COPY script.sh .
COPY final.py .
COPY requirements.in .

RUN pip3 install --compile --no-cache-dir -r requirements.in

RUN chmod +x script.sh
CMD ./script.sh
