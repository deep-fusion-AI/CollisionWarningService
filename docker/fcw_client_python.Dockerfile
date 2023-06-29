FROM python:3.8-slim

RUN apt-get update \
    && apt-get install -y python3-pip git

RUN python -m pip install --upgrade pip

RUN mkdir -p /root/opencv

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Prague

RUN cd /root/opencv \
    && pip3 install *.whl

RUN cd /root/ \
    && git clone https://github.com/klepo/era-5g-client.git

RUN cd /root/era-5g-client \
    && pip3 install -r requirements.txt \
    && pip3 install .

ENTRYPOINT ["/root/fcw_client_python_start.sh"]

COPY fcw/client_python/ /root/fcw/client_python

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

COPY docker/fcw_client_python_start.sh /root/fcw_client_python_start.sh

RUN chmod +x /root/fcw_client_python_start.sh

ENV NETAPP_PORT=5897
    
EXPOSE 5897
