import argparse
import logging
import sys
import time
from typing import Optional, Dict, Any
import av
from av.container.output import OutputContainer
from av.stream import Stream
import cv2
import numpy
import zmq
from zmq import Socket, Context

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("FCW visualization")

from fcw_core.vizualization import *


def recv_array(socket: Socket, flags=0, copy=True, track=False) -> (Dict[str, Any], np.ndarray):
    """recv a numpy array"""
    try:
        md = socket.recv_json(flags=flags)
        msg = socket.recv(flags=flags, copy=copy, track=track)
        buf = memoryview(msg)
        image = numpy.frombuffer(buf, dtype=md['dtype'])
        return md['results'], image.reshape(md['shape'])
    except zmq.error.Again as e:
        logger.debug("Missing visualization data!")
        return {}, None


def mark_vehicles(
    size: tuple, objects: list, camera: Camera, marker: Image, anchor: tuple = (0, 0), to_rectified=False
):
    image = Image.new("RGBA", size, color=(0, 0, 0, 0))
    ax, ay = anchor
    # loc (N,2) xy
    for o in objects:
        X = np.atleast_2d([o["location"][0], o["location"][1], 0])
        scr_loc, _ = camera.project_points(X, near=1, to_rectified=to_rectified)
        if scr_loc.shape[0] > 0:
            x, y = scr_loc[0]
            image.paste(marker, (int(x - ax), int(y - ay)))
    return image


def draw_world_objects(
    size: tuple,
    camera: Camera,
    objects: list,
    to_rectified=False,
):
    image = Image.new("RGBA", size)
    draw = ImageDraw.Draw(image)

    for o in objects:
        X = np.atleast_2d([o["location"][0], o["location"][1], 0])
        scr_loc, _ = camera.project_points(X, to_rectified=to_rectified)
        if scr_loc.size > 0:
            x, y = scr_loc[0]
            draw.line([(x - 10, y), (x + 10, y)], fill=(255, 255, 0, 128), width=3)
            draw.line([(x, y - 10), (x, y + 10)], fill=(255, 255, 0, 128), width=3)

        X = np.array(o["path"])
        n = X.shape[0]
        X = np.hstack([X, np.zeros((n, 1))])
        scr_loc, _ = camera.project_points(X, near=5, to_rectified=to_rectified)
        scr_loc = list(map(tuple, scr_loc))
        draw.line(scr_loc, fill=(0, 255, 0, 255), width=1)

    return image


def draw_image_trackers(
    size: tuple,
    trackers: list,
):
    image = Image.new("RGBA", size)
    draw = ImageDraw.Draw(image)

    for t in trackers:
        x1, y1, x2, y2 = t["bbox"]
        color = (0, 255, 0, 64)
        outline = (0, 255, 0, 128)
        if t["age"] < 3 or t["hit_streak"] == 0:  # TODO: call it.is_reliable()
            color = (255, 255, 0, 32)
            outline = None
        draw.rectangle((x1, y1, x2, y2), fill=color, outline=outline)

    return image


def main(args=None):
    parser = argparse.ArgumentParser(description='Visualization of Forward Collision Warning Service')
    parser.add_argument("-z", "--zmq_port", type=str, help="ZeroMQ port", default="5558")
    parser.add_argument("-u", "--rtsp_port", type=str, help="RTSP port, address is rtsp://localhost:{rtsp_port}/video", default="8554")
    args = parser.parse_args()

    context: Context = zmq.Context()
    socket: Socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:%s" % args.zmq_port)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVTIMEO, 2000)

    camera: Optional[Camera] = None
    config: Optional[Dict] = None
    output = None

    while True:
        try:
            results, image = recv_array(socket)
            if results is None or image is None:
                cv2.destroyAllWindows()
                continue
            if not config or config != results["config"]:
                config = results["config"]
                if "camera_config" not in config:
                    config = None
                    continue
                logger.info("Initializing camera calibration")
                camera = Camera.from_dict(config["camera_config"])
                logo = cog_logo((64, 64))
                coord_sys = draw_world_coordinate_system(camera.rectified_size, camera)
                coord_sys.putalpha(64)
                if type(config["config"]["fcw"].get("danger_zone")) == dict:
                    zone = Polygon(list(config["config"]["fcw"].get("danger_zone").values()))
                else:
                    zone = Polygon(config["config"]["fcw"].get("danger_zone"))
                danger_zone = draw_danger_zone(camera.rectified_size, camera, zone)
                horizon = draw_horizon(camera.rectified_size, camera, width=1, fill=(255, 255, 0, 64))
                marker, marker_anchor = vehicle_marker_image(scale=3)

                if output is not None:
                    output.close()
                #TODO: Check RTSP server is running
                output: OutputContainer = av.open(
                    f"rtsp://localhost:{args.rtsp_port}/video", mode="w", format="rtsp", options={'rtsp_transport': 'tcp'}, timeout=2
                )
                output.flags |= output.flags.NONBLOCK
                out_stream: Stream = output.add_stream("h264", 30)
                logger.debug(out_stream.codec_context.is_open)
                out_stream.pix_fmt = "yuv420p"
                out_stream.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "0"}
                out_stream.width = image.shape[1]
                out_stream.height = image.shape[0]

            logger.debug(results["dangerous_detections"])
            logger.debug(results["objects"])
            logger.debug("--------")

            base_undistorted = Image.fromarray(image[..., ::-1], "RGB").convert("RGBA")
            # Layers showing various information
            sz = base_undistorted.size
            layers = [
                (coord_sys, None),
                (danger_zone, None),
                (horizon, None),
                (draw_image_trackers(sz, list(results["dangerous_detections"].values())), None),
                (draw_world_objects(sz, camera, list(results["objects"]), to_rectified=True), None),
            ]
            # Compose layers together
            compose_layers(base_undistorted, *layers)
            object_statuses: List[ObjectStatus] = []
            for object_status_str in results["objects"]:
                object_status = ObjectStatus(
                    id=object_status_str["id"],
                    distance=object_status_str["distance"],
                    location=object_status_str["location"],
                    path=object_status_str["path"],
                    is_in_danger_zone=object_status_str["is_in_danger_zone"],
                    crosses_danger_zone=object_status_str["crosses_danger_zone"],
                    time_to_collision=object_status_str["time_to_collision"]
                )
                object_statuses.append(object_status)
            w1, h1 = base_undistorted.size
            compose_layers(
                base_undistorted,  # Original image
                (tracking_info((w1, 16), object_statuses), (0, 0)),
                (mark_vehicles(
                    camera.image_size,
                    list(results["objects"]),
                    camera,
                    marker,
                    marker_anchor,
                    to_rectified=True
                ), (0, 0)),
                (logo, (8, 16 + 8)),
            )
            # Convert to OpenCV for display
            cv_image = np.array(base_undistorted.convert("RGB"))[..., ::-1]

            try:
                out_frame = av.VideoFrame.from_ndarray(cv_image, format="bgr24")
                out_packet = out_stream.encode(out_frame)
                logger.info(out_packet)
                output.mux(out_packet)
                # time.sleep(1 / float(in_stream.framerate))
            except Exception as ex:
                config = None
                logger.error(repr(ex))
            #try:
                # Display the image
            #    cv2.imshow("FCW", cv_image)
            #    cv2.waitKey(1)
            #except Exception as ex:
            #    logger.debug(repr(ex))
        except KeyboardInterrupt:
            logger.info("Terminating ...")
            break


if __name__ == '__main__':
    main()
