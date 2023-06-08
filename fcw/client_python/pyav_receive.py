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
from fractions import Fraction

from socketserver import StreamRequestHandler, TCPServer

from av import Packet
from av.video.codeccontext import VideoCodecContext
from av.codec import CodecContext


class DumpHandler(StreamRequestHandler):
    def handle(self) -> None:

        decoder: VideoCodecContext = CodecContext.create('h264', 'r')
        decoder.width = 0
        decoder.height = 0
        decoder.pix_fmt = 'yuv420p'

        """receive json packets from client"""
        print('connection from {}:{}'.format(*self.client_address))
        try:
            while True:
                data = self.rfile.readline()
                if not data:
                    break
                packet_json = json.loads(data.decode().rstrip())
                packet_data = base64.b64decode(packet_json["bytes"].encode("utf-8"))
                packet = Packet(packet_data)
                packet.dts = packet_json["dts"]
                packet.pts = packet_json["pts"]
                packet.time_base = Fraction(packet_json["time_base_numerator"], packet_json["time_base_denominator"])
                if decoder.width != packet_json["width"] or decoder.height != packet_json["height"]:
                    decoder.width = packet_json["width"]
                    decoder.height = packet_json["height"]

                print(f"time diff: {(time.time_ns() - packet_json['timestamp']) * 1.0e-9:.3f}s")
                frame = []
                try:
                    frame = decoder.decode(packet)
                except Exception as e:
                    print(e)

                if len(frame) > 0:
                    frame[0].to_image().save('output/frame-%04d.jpg' % packet.dts)
                    print(frame[0])
        finally:
            print('disconnected from {}:{}'.format(*self.client_address))


server_address = ('147.229.13.96', 9090)
print('starting up on {}:{}'.format(*server_address))
with TCPServer(server_address, DumpHandler) as server:
    print('waiting for a connection')
    server.serve_forever()
