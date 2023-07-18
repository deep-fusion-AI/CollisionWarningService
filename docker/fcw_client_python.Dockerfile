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

COPY fcw/client_python/ /root/fcw/client_python
COPY fcw/core/ /root/fcw/core
COPY data /root/data
COPY config /root/config
COPY videos /root/videos
COPY pyproject.toml /root/
COPY poetry.lock /root/
COPY README.md /root/

RUN pip3 install poetry

RUN cd /root/ \
    && poetry config virtualenvs.create false \
    && poetry install

ENTRYPOINT ["/root/fcw_client_python_start.sh"]

COPY docker/fcw_client_python_start.sh /root/fcw_client_python_start.sh

RUN chmod +x /root/fcw_client_python_start.sh

ENV NETAPP_ADDRESS=http://127.0.0.1:5897

EXPOSE 5897
