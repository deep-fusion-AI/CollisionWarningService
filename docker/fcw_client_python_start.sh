#!/bin/bash

cd
exec poetry run fcw_client_python -c config/config.yaml --camera videos/video3.yaml videos/video3.mp4
