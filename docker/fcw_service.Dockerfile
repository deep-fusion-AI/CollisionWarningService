FROM python:3.8-slim

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

RUN apt-get update \
    && apt-get install -y \
    git \
    python3-pip \
    python-is-python3 \
    build-essential \
    cmake \
    gcc \
    ffmpeg

RUN python3 -m pip install --upgrade pip

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Prague

COPY fcw-service/ /root/fcw-service
COPY fcw-core/ /root/fcw-core
COPY fcw-core-utils/ /root/fcw-core-utils

RUN pip3 install poetry

RUN cd /root/fcw-service \
    && poetry config virtualenvs.create false \
    && poetry install

ENTRYPOINT ["/root/fcw_service_start.sh"]

COPY docker/fcw_service_start.sh /root/fcw_service_start.sh

RUN chmod +x /root/fcw_service_start.sh

ENV NETAPP_PORT=5896
    
EXPOSE 5896
