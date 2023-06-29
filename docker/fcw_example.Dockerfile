FROM python:3.8-slim

RUN apt-get update \
    && apt-get install -y python3 python-is-python3 python3-pip git

RUN python3 -m pip install --upgrade pip

#RUN pip install torch===1.13.1+cu117 torchvision===0.14.1+cu117 torchaudio===0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

RUN mkdir -p /root/opencv

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Prague

RUN cd /root/opencv \
    && pip3 install *.whl

ENTRYPOINT ["/root/fcw_example_start.sh"]

COPY fcw/core/ /root/fcw/core

RUN cd /root/fcw/core \
    && pip3 install -r requirements.txt

COPY data /root/data
COPY config /root/config
COPY videos /root/videos

COPY pyproject.toml /root/
COPY README.md /root/

RUN pip3 install poetry

#export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
RUN cd /root/ \
    && poetry config virtualenvs.create false \
    && poetry install

COPY docker/fcw_example_start.sh /root/fcw_example_start.sh

RUN chmod +x /root/fcw_example_start.sh

EXPOSE 5897