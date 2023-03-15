import secrets
from queue import Queue
import cv2
import argparse
import os
import numpy as np
import flask_socketio
from flask import Flask, Response, request, session
from flask_session import Session
import logging

from era_5g_interface.task_handler_gstreamer_internal_q import \
    TaskHandlerGstreamerInternalQ, TaskHandlerGstreamer
from era_5g_interface.task_handler_internal_q import TaskHandlerInternalQ

from collision_worker import CollisionWorker

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

# list of registered tasks
tasks = dict()

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
    # print(f"register {session.sid} {session}")
    # print(f"{request}")
    args = request.get_json(silent=True)
    print(args)
    gstreamer = False
    config = {}
    camera_config = {}
    fps = 30
    if args:
        gstreamer = args.get("gstreamer", False)
        config = args.get("config", {})
        camera_config = args.get("camera_config", {})
        fps = args.get("fps", 30)
        print(f"Config: {config}")
        print(f"Camera config: {camera_config}")
        print(f"FPS: {fps}")

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

    tasks[session.sid] = {"task_handler": task,
                          "worker": worker}

    print(f"Client registered: {session.sid}")
    if gstreamer:
        return {"port": port}, 200
    else:
        return Response(status=204)


@app.route('/unregister', methods=['POST'])
def unregister():
    """Disconnects the websocket and removes the task from the memory.

    Returns:
        _type_: 204 status
    """
    session_id = session.sid
    # print(f"unregister {session.sid} {session}")
    # print(f"{request}")
    if session.pop('registered', None):
        task_and_worker = tasks.pop(session.sid)
        task = task_and_worker["task_handler"]
        task_and_worker["worker"].stop()
        task_and_worker["task_handler"].stop()
        flask_socketio.disconnect(task.websocket_id, namespace="/results")
        if isinstance(task, TaskHandlerGstreamer):
            free_ports.append(task.port)
        print(f"Client unregistered: {session_id}")

    return Response(status=204)


@app.route('/image', methods=['POST'])
def post_image():
    """Allows to receive jpg-encoded image using the HTTP transport"""

    sid = session.sid
    task = tasks[sid]["task_handler"]

    if "timestamps[]" in request.args:
        timestamps = request.args.to_dict(flat=False)['timestamps[]']
    else:
        timestamps = []
    # convert string of image data to uint8
    index = 0
    for file in request.files.to_dict(flat=False)['files']:
        # print(file)
        nparr = np.frombuffer(file.read(), np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # store the image to the appropriate task
        task.store_image({"sid": sid, "websocket_id": task.websocket_id, "timestamp": timestamps[index]}, img)
        index += 1
    return Response(status=204)


@socketio.on('connect', namespace='/results')
def connect(auth):
    """Creates a websocket connection to the client for passing the results.

    Raises:
        ConnectionRefusedError: Raised when attempt for connection were made
            without registering first.
    """

    # print(f"connect {session.sid} {session}")
    # print(f"{request.sid} {request}")
    if 'registered' not in session:
        # TODO: disconnect?
        # flask_socketio.disconnect(request.sid, namespace="/results")
        raise ConnectionRefusedError('Need to call /register first.')

    sid = request.sid
    print(f"Client connected: session id: {session.sid}, websocket id: {sid}")
    tasks[session.sid]["worker"].start()
    tasks[session.sid]["task_handler"].websocket_id = sid
    tasks[session.sid]["task_handler"].start()
    flask_socketio.send("You are connected", namespace='/results', to=sid)


@socketio.on('disconnect', namespace='/results')
def disconnect():
    # print(f"disconnect {session.sid} {session}")
    # print(f"{request.sid} {request}")
    print(f"Client disconnected: session id: {session.sid}, websocket id: {request.sid}")


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
        print("Port range specified in wrong format. The correct format is port_start:port_end, e.g. 5001:5003.")
        exit()

    # runs the flask server
    # allow_unsafe_werkzeug needs to be true to run inside the docker
    # TODO: use better webserver
    socketio.run(app, port=NETAPP_PORT, host='0.0.0.0', allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
