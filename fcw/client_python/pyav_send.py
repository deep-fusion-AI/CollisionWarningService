from ipywidgets import interact
from matplotlib import pyplot as plt
import ffmpeg
import ipywidgets as widgets
import numpy as np
import time
import cv2
import socket
import struct
import json
import base64
import random

import av
from av.stream import Stream
from av import Packet
from av.container import InputContainer
from av.video.codeccontext import VideoCodecContext
from av.codec import CodecContext


clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
clientSocket.connect(("147.229.13.96", 9090))


input_file = "../../videos/video3.mp4"

width = 962
height = 720
num_frames = 100
fps = 30
#
input_container: InputContainer = av.open(input_file)
input_stream: Stream = input_container.streams.video[0]

encoder: VideoCodecContext = CodecContext.create('h264', 'w')
#encoder.width = width
#encoder.height = height
encoder.width = input_stream.codec_context.width
encoder.height = input_stream.codec_context.height
#encoder.pix_fmt = "yuv420p"
encoder.pix_fmt = input_stream.codec_context.pix_fmt
#encoder.framerate = fps
encoder.framerate = input_stream.guessed_rate
encoder.options = {"crf": "0", "preset": "ultrafast", "tune": "zerolatency", "byte-stream": "true"}

duration = 10 ** 9 / (fps / 1)
pts = 0

frame_num = 0
for frame in input_container.decode(input_stream):
    img_frame = frame.to_image()
    out_frame = av.VideoFrame.from_image(img_frame)

    out_frame.pts = time.time_ns()
    for packet in encoder.encode(out_frame):
        img_frame.save('input/frame-%04d.jpg' % packet.dts)
        print(packet)
        #packet_size = packet.size
        #packet_size_as_4_bytes = struct.pack('>I', packet_size)
        #clientSocket.send(packet_size_as_4_bytes)
        #if bool(random.getrandbits(1)):
        #    continue
        message = (json.dumps({"dts": packet.dts,
                               "pts": packet.pts,
                               "width": encoder.width,
                               "height": encoder.height,
                               "pix_fmt": encoder.pix_fmt,
                               "is_keyframe": packet.is_keyframe,
                               "timestamp": time.time_ns(),
                               "time_base_numerator": packet.time_base.numerator,
                               "time_base_denominator": packet.time_base.denominator,
                               "bytes": base64.b64encode(bytes(packet)).decode("utf-8")}) + '\n').encode()
        #print(message)
        clientSocket.sendall(message)

    frame_num += 1

