FROM nvidia/cuda:12.3.1-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Prague

RUN apt-get update \
    && apt-get install -y \
    tzdata \
    git \
    python3-pip \
    python-is-python3 \
    build-essential \
    cmake \
    gcc \
    ffmpeg

RUN pip install --upgrade pip

COPY fcw-service/ /root/fcw-service
COPY fcw-core/ /root/fcw-core
COPY fcw-core-utils/ /root/fcw-core-utils

RUN pip install poetry

RUN cd /root/fcw-core-utils \
    && poetry config virtualenvs.create false \
    && poetry install

RUN cd /root/fcw-core \
    && poetry config virtualenvs.create false \
    && poetry install

RUN cd /root/fcw-service \
    && poetry config virtualenvs.create false \
    && poetry install

COPY docker/yolov5m6.pt /root/fcw-service/yolov5m6.pt
COPY docker/yolov5n6.pt /root/fcw-service/yolov5n6.pt
RUN git clone https://github.com/ultralytics/yolov5  # clone
RUN cd yolov5 && git checkout tags/v7.0
RUN cd yolov5 && pip3 install -r requirements.txt  # install
RUN mkdir -p /root/.cache/torch/hub/ultralytics_yolov5_master
RUN cp -r yolov5/* /root/.cache/torch/hub/ultralytics_yolov5_master

COPY docker/fcw_service_start.sh /root/fcw_service_start.sh
COPY data/ /root/data

#COPY --from=0 /mediamtx /root/mediamtx
#COPY docker/mediamtx.yml /root/mediamtx.yml

ENTRYPOINT ["/root/fcw_service_start.sh"]

RUN chmod +x /root/fcw_service_start.sh

ENV NETAPP_PORT=5896
    
EXPOSE 5896
#EXPOSE 8554
EXPOSE 5558
