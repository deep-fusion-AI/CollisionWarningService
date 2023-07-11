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

RUN cd /root/ \
    && git clone https://github.com/klepo/Reference-NetApp.git

RUN cd /root/Reference-NetApp/src/python/era_5g_object_detection_common  \
    && pip3 install -r requirements.txt \
    && pip3 install .

RUN cd /root/Reference-NetApp/src/python/era_5g_object_detection_standalone \
    && pip3 install -r requirements.txt \
    && pip3 install . 

RUN cd /root/ \
    && git clone https://github.com/klepo/era-5g-interface.git

RUN cd /root/era-5g-interface \
    && pip3 install -r requirements.txt \
    && pip3 install .

COPY fcw/core/ /root/fcw/core

RUN cd /root/fcw/core \
    && pip3 install -r requirements.txt

COPY fcw/service/ /root/fcw/service

RUN cd /root/fcw/service \
    && pip3 install -r requirements.txt

COPY pyproject.toml /root/
COPY poetry.lock /root/
COPY README.md /root/

RUN pip3 install poetry

RUN cd /root/ \
    && poetry config virtualenvs.create false \
    && poetry install

ENTRYPOINT ["/root/fcw_service_start.sh"]

COPY docker/fcw_service_start.sh /root/fcw_service_start.sh

RUN chmod +x /root/fcw_service_start.sh

ENV NETAPP_PORT=5897
    
EXPOSE 5897
