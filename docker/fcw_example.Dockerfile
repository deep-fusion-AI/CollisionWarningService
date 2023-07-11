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

COPY fcw/core/ /root/fcw/core

RUN cd /root/fcw/core \
    && pip3 install -r requirements.txt

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

ENTRYPOINT ["/root/fcw_example_start.sh"]

COPY docker/fcw_example_start.sh /root/fcw_example_start.sh

RUN chmod +x /root/fcw_example_start.sh

EXPOSE 5897