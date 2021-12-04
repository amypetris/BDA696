FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y mariadb-client

COPY rolling_avg.sql .
COPY baseball.sql .
COPY script.sh .

RUN chmod +x script.sh
CMD ./script.sh
