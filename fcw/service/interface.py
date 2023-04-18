import argparse
import base64
import binascii
import logging
import os
import secrets
import time
from dataclasses import dataclass
from queue import Queue
from typing import Dict

import cv2
import flask_socketio
import numpy as np
from flask import Flask, Response, request, session
from flask_session import Session

from fcw.service.collision_worker import CollisionWorker
from era_5g_interface.task_handler import TaskHandler
from era_5g_interface.task_handler_gstreamer_internal_q import \
    TaskHandlerGstreamerInternalQ, TaskHandlerGstreamer
from era_5g_interface.task_handler_internal_q import TaskHandlerInternalQ

# port of the netapp's server
NETAPP_PORT = os.getenv("NETAPP_PORT", 5896)

# flask initialization
app = Flask(__name__)
app.secret_key = secrets.token_hex()
app.config['SESSION_TYPE'] = 'filesystem'

# socketio initialization for sending results to the client
Session(app)
socketio = flask_socketio.SocketIO(app, manage_session=False, async_mode='threading')

# list of available ports for gstreamer communication - they need to be exposed when running in docker / k8s
# could be changed using the script arguments
free_ports = [5001, 5002, 5003]


@dataclass
class TaskAndWorker:
    task: TaskHandler
    worker: CollisionWorker


# list of registered tasks
tasks: Dict[str, TaskAndWorker] = dict()

# the image detector to be used
detector_thread = None


class ArgFormatError(Exception):
    pass


@app.route('/register', methods=['POST'])
def register():
    """Needs to be called before an attempt to open WS is made.

    Returns:
        _type_: The port used for gstreamer communication.
    """

    args = request.get_json(silent=True)
    gstreamer = False
    config = {}
    camera_config = {}
    fps = 30
    if args:
        gstreamer = args.get("gstreamer", False)
        config = args.get("config", {})
        camera_config = args.get("camera_config", {})
        fps = args.get("fps", 30)
        logging.info(f"Config: {config}")
        logging.info(f"Camera config: {camera_config}")
        logging.info(f"FPS: {fps}")

    if gstreamer and not free_ports:
        return {"error": "Not enough resources"}, 503

    session['registered'] = True

    # queue with received images
    # TODO: adjust the queue length
    image_queue = Queue(30)

    # select the appropriate task handler, depends on whether the client wants to use
    # gstreamer to pass the images or not
    if gstreamer:
        port = free_ports.pop(0)
        task = TaskHandlerGstreamerInternalQ(session.sid, port, image_queue, daemon=True)
    else:
        task = TaskHandlerInternalQ(session.sid, image_queue, daemon=True)
    # Create worker
    worker = CollisionWorker(
        image_queue, app,
        config, camera_config, fps,
        name="Detector",
        daemon=True
    )

    tasks[session.sid] = TaskAndWorker(task, worker)

    logging.info(f"Client registered: {session.sid}")
    if gstreamer:
        return {"port": port}, 200
    else:
        return Response(status=204)


@app.route('/unregister', methods=['POST'])
def unregister():
    """Disconnects the websocket and removes the task from the memory.

    Returns:
        Response: 204 status
    """

    session_id = session.sid
    if session.pop('registered', None):
        task_and_worker = tasks.pop(session.sid)
        task = task_and_worker.task
        task_and_worker.worker.stop()
        task_and_worker.task.stop()
        flask_socketio.disconnect(task.websocket_id, namespace="/results")
        if isinstance(task, TaskHandlerGstreamer):
            free_ports.append(task.port)
        logging.info(f"Client unregistered: {session_id}")

    return Response(status=204)


@app.route('/image', methods=['POST'])
def image_callback_http():
    """Allows to receive jpg-encoded image using the HTTP transport"""

    recv_timestamp = time.time_ns()

    sid = session.sid
    task = tasks[sid].task

    if "timestamps[]" in request.args:
        timestamps = request.args.to_dict(flat=False)['timestamps[]']
    else:
        timestamps = []
    # convert string of image data to uint8
    index = 0
    for file in request.files.to_dict(flat=False)['files']:
        ndarray = np.frombuffer(file.read(), np.uint8)
        # decode image
        img = cv2.imdecode(ndarray, cv2.IMREAD_COLOR)

        # store the image to the appropriate task
        task.store_image(
            {"sid": sid,
             "websocket_id": task.websocket_id,
             "timestamp": timestamps[index],
             "recv_timestamp": recv_timestamp},
            img
        )
        index += 1
    return Response(status=204)


@socketio.on('image', namespace='/data')
def image_callback_websocket(data: dict):
    """Allows to receive jpg-encoded image using the websocket transport

    Args:
        data (dict): A base64 encoded image frame and (optionally) related timestamp in format:
            {'frame': 'base64data', 'timestamp': 'int'}

    Raises:
        ConnectionRefusedError: Raised when attempt for connection were made
            without registering first or frame was not passed in correct format.
    """

    logging.debug("A frame received using ws")
    recv_timestamp = time.time_ns()

    if 'timestamp' in data:
        timestamp = data['timestamp']
    else:
        logging.debug("Timestamp not set, setting default value")
        timestamp = 0
    if 'registered' not in session:
        logging.error(f"Non-registered client tried to send data")
        flask_socketio.emit(
            "image_error",
            {"timestamp": timestamp,
             "error": "Need to call /register first."},
            namespace='/data',
            to=request.sid
        )
        return
    if 'frame' not in data:
        logging.error(f"Data does not contain frame.")
        flask_socketio.emit(
            "image_error",
            {"timestamp": timestamp,
             "error": "Data does not contain frame."},
            namespace='/data',
            to=request.sid
        )
        return

    sid = session.sid
    task = tasks[sid].task

    try:
        frame = base64.b64decode(data["frame"])
        task.store_image(
            {"sid": session.sid,
             "websocket_id": task.websocket_id,
             "timestamp": timestamp,
             "recv_timestamp": recv_timestamp,
             "decoded": False},
            np.frombuffer(frame, dtype=np.uint8)
        )
    except (ValueError, binascii.Error) as error:
        logging.error(f"Failed to decode frame data: {error}")
        flask_socketio.emit(
            "image_error",
            {"timestamp": timestamp,
             "error": f"Failed to decode frame data: {error}"},
            namespace='/data',
            to=request.sid
        )


@socketio.on('connect', namespace='/results')
def connect_results(auth):
    """Creates a websocket connection to the client for passing the results.

    Raises:
        ConnectionRefusedError: Raised when attempt for connection were made
            without registering first.
    """

    if 'registered' not in session:
        # TODO: disconnect?
        # flask_socketio.disconnect(request.sid, namespace="/results")
        logging.error(f"Need to call /register first. Session id: {session.sid}, ws_sid: {request.sid}")
        raise ConnectionRefusedError('Need to call /register first.')

    sid = request.sid
    logging.info(f"Client connected: session id: {session.sid}, websocket id: {sid}")
    tasks[session.sid].worker.start()
    tasks[session.sid].task.websocket_id = sid
    tasks[session.sid].task.start()
    t0 = time.time_ns()
    while True:
        if tasks[session.sid].worker.is_alive():
            break
        if time.time_ns() > t0 + 5 * 1.0e+9:
            logging.error(f"Timed out to start worker. Session id: {session.sid}, ws_sid: {request.sid}")
            raise ConnectionRefusedError('Timed out to start worker.')
    t0 = time.time_ns()
    while True:
        if tasks[session.sid].task.is_alive():
            break
        if time.time_ns() > t0 + 5 * 1.0e+9:
            logging.error(f"Timed out to start task. Session id: {session.sid}, ws_sid: {request.sid}")
            raise ConnectionRefusedError('Timed out to start task.')
    # TODO: Check task is running, Gstreamer capture can failed
    flask_socketio.send("You are connected", namespace='/results', to=sid)


@socketio.on('connect', namespace='/data')
def connect_data(auth):
    """Creates a websocket connection to the client for passing the data.

    Raises:
        ConnectionRefusedError: Raised when attempt for connection were made
            without registering first.
    """

    if 'registered' not in session:
        logging.error(f"Need to call /register first. Session id: {session.sid}, ws_sid: {request.sid}")
        raise ConnectionRefusedError('Need to call /register first.')

    logging.info(f"Connected data. Session id: {session.sid}, ws_sid: {request.sid}")
    flask_socketio.send("You are connected", namespace='/data', to=request.sid)


@socketio.on('disconnect', namespace='/results')
def disconnect_results():
    logging.info(f"Client disconnected from /results namespace: session id: {session.sid}, websocket id: {request.sid}")


@socketio.on('disconnect', namespace='/data')
def disconnect_data():
    logging.info(f"Client disconnected from /data namespace: session id: {session.sid}, websocket id: {request.sid}")


def get_ports_range(ports_range):
    if ports_range.count(':') != 1:
        raise ArgFormatError
    r1, r2 = ports_range.split(':')
    if int(r2) <= int(r1):
        raise ArgFormatError
    return [port for port in range(int(r1), int(r2) + 1)]


def main(args=None):
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Standalone variant of object detection NetApp')
    parser.add_argument(
        '--ports',
        default="5001:5003",
        help="Specify the range of ports available for gstreamer connections. Format "
             "port_start:port_end. Default is 5001:5003."
    )
    args = parser.parse_args()
    global free_ports
    try:
        free_ports = get_ports_range(args.ports)
    except ArgFormatError:
        logging.error("Port range specified in wrong format. The correct format is port_start:port_end, e.g. 5001:5003.")
        exit()

    # runs the flask server
    # allow_unsafe_werkzeug needs to be true to run inside the docker
    # TODO: use better webserver
    socketio.run(app, port=NETAPP_PORT, host='0.0.0.0', allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
