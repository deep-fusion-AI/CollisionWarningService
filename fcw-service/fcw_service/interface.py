import os
import numpy as np
import time
from dataclasses import dataclass
from queue import Queue
from typing import Dict, Tuple, Any
import threading
import logging
import sys
import traceback

from fcw_service.collision_worker import CollisionWorker
from fcw_core.yolo_detector import YOLODetector

from era_5g_interface.task_handler_internal_q import TaskHandlerInternalQ
import era_5g_interface.interface_helpers
from era_5g_interface.interface_helpers import HeartBeatSender
from era_5g_interface.dataclasses.control_command import ControlCommand, ControlCmdType

from era_5g_interface.channels import CallbackInfoServer, ChannelType, DATA_NAMESPACE, DATA_ERROR_EVENT
from era_5g_server.server import NetworkApplicationServer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("FCW interface")

# Port of the 5G-ERA Network Application's server.
NETAPP_PORT = os.getenv("NETAPP_PORT", 5896)
# Input queue size.
NETAPP_INPUT_QUEUE = int(os.getenv("NETAPP_INPUT_QUEUE", 1))
# Event name for image error.
IMAGE_ERROR_EVENT = str("image_error")


@dataclass
class TaskAndWorker:
    """Class for task and worker."""

    task: TaskHandlerInternalQ
    worker: CollisionWorker


class Server(NetworkApplicationServer):
    """FCW server receives images from clients, manages tasks and workers, sends results to clients."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Constructor.

        Args:
            *args: NetworkApplicationServer arguments.
            **kwargs: NetworkApplicationServer arguments.
        """

        super().__init__(
            callbacks_info={"image": CallbackInfoServer(ChannelType.H264, self.image_callback)}, *args,
            **kwargs
        )

        # List of registered tasks.
        self.tasks: Dict[str, TaskAndWorker] = dict()

        self.heart_beat_sender = HeartBeatSender()
        self.heart_beat_timer()

    def heart_beat_timer(self):
        """Heart beat timer."""

        latencies = []
        for task_and_worker in self.tasks.values():
            latencies.extend(task_and_worker.worker.latency_measurements.get_latencies())
        avg_latency = 0
        if len(latencies) > 0:
            avg_latency = float(np.mean(np.array(latencies)))

        queue_size = 1
        queue_occupancy = 1

        self.heart_beat_sender.send_middleware_heart_beat(
            avg_latency=avg_latency,
            queue_size=queue_size,
            queue_occupancy=queue_occupancy,
            current_robot_count=len(self.tasks),
        )
        threading.Timer(era_5g_interface.interface_helpers.MIDDLEWARE_REPORT_INTERVAL, self.heart_beat_timer).start()

    def image_callback(self, sid: str, data: Dict[str, Any]) -> None:
        """Allows to receive decoded image using the websocket transport.

        Args:
            sid (str): Namespace sid.
            data (Dict[str, Any]): Data dict including decoded frame (data["frame"]) and send timestamp
                (data["timestamp"]).
        """

        eio_sid = self._sio.manager.eio_sid_from_sid(sid, DATA_NAMESPACE)

        if eio_sid not in self.tasks:
            logger.error(f"Non-registered client {eio_sid} tried to send data")
            self.send_data({"message": "Non-registered client tried to send data"}, DATA_ERROR_EVENT, sid)
            return

        task = self.tasks[eio_sid].task
        task.store_data({"timestamp": data["timestamp"], "recv_timestamp": time.perf_counter_ns()}, data["frame"])

    def command_callback(self, command: ControlCommand, sid: str) -> Tuple[bool, str]:
        """Process initialization control command - create task, worker and start the worker.

        Args:
            command (ControlCommand): Control command to be processed.
            sid (str): Namespace sid.

        Returns:
            (initialized (bool), message (str)): If False, initialization failed.
        """

        eio_sid = self.get_eio_sid_of_control(sid)

        logger.info(f"Control command {command} processing: session id: {sid}")

        if command and command.cmd_type == ControlCmdType.INIT:
            # Check that initialization has not been called before.
            if eio_sid in self.tasks:
                logger.error(f"Client attempted to call initialization multiple times")
                self.send_command_error("Initialization has already been called before", sid)
                return False, "Initialization has already been called before"

            args = command.data
            config = {}
            camera_config = {}
            fps = 30
            viz = True
            viz_zmq_port = 5558
            if args:
                config = args.get("config", config)
                camera_config = args.get("camera_config", camera_config)
                fps = args.get("fps", fps)
                viz = args.get("viz", viz)
                viz_zmq_port = args.get("viz_zmq_port", viz_zmq_port)
                logger.info(f"Config: {config}")
                logger.info(f"Camera config: {camera_config}")
                logger.info(f"ZeroMQ visualization: {viz}, port: {viz_zmq_port}")

            # Queue with received images.
            image_queue = Queue(NETAPP_INPUT_QUEUE)

            task = TaskHandlerInternalQ(image_queue)

            try:
                # Create worker.
                worker = CollisionWorker(
                    image_queue,
                    lambda results: self.send_data(data=results, event="results", sid=self.get_sid_of_data(eio_sid)),
                    config,
                    camera_config,
                    fps,
                    viz,
                    viz_zmq_port,
                    name=f"Detector {eio_sid}",
                    daemon=True,
                )
            except Exception as ex:
                logger.error(f"Failed to create CollisionWorker: {repr(ex)}")
                logger.error(traceback.format_exc())
                self.send_command_error(f"Failed to create CollisionWorker: {repr(ex)}", sid)
                return False, f"Failed to create CollisionWorker: {repr(ex)}"

            self.tasks[eio_sid] = TaskAndWorker(task, worker)
            self.tasks[eio_sid].worker.start()
            t0 = time.perf_counter_ns()
            while True:
                if self.tasks[eio_sid].worker.is_alive():
                    break
                if time.perf_counter_ns() > t0 + 5 * 1.0e9:
                    logger.error(f"Timed out to start worker, eio_sid {eio_sid}, sid {sid}")
                    return False, f"Timed out to start worker"

            logger.info(f"Task handler and worker created and started: {eio_sid}")

        logger.info(
            f"Control command applied, eio_sid {eio_sid}, sid {sid}, "
            f"results sid {self.get_sid_of_data(eio_sid)}, command {command}"
        )
        return True, (f"Control command applied, eio_sid {eio_sid}, sid {sid}, results sid"
                      f" {self.get_sid_of_data(eio_sid)}, command {command}")

    def disconnect_callback(self, sid: str) -> None:
        """Called with client disconnection - deletes task and worker.

        Args:
            sid (str): Namespace sid.
        """

        eio_sid = self.get_eio_sid_of_data(sid)
        task_and_worker = self.tasks.get(eio_sid)
        if task_and_worker:
            task_and_worker.worker.stop()
            task_and_worker.worker.join()
            del self.tasks[eio_sid]
            del task_and_worker
            logger.info(f"Task handler and worker deleted: {eio_sid}")

        logger.info(f"Client disconnected from {DATA_NAMESPACE} namespace, eio_sid {eio_sid}, sid {sid}")


def main():
    """Main function."""

    # parser = argparse.ArgumentParser(description='Forward Collision Warning Service')
    # args = parser.parse_args()

    logger.info(f"The size of the queue set to: {NETAPP_INPUT_QUEUE}")

    logger.info("Initializing default object detector for faster first startup")
    detector = YOLODetector.from_dict({})
    del detector

    server = Server(port=NETAPP_PORT, host="0.0.0.0")

    try:
        server.run_server()
    except KeyboardInterrupt:
        logger.info("Terminating ...")


if __name__ == "__main__":
    main()
