import json
import os
import time
from dataclasses import asdict
from queue import Queue
from typing import List

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Publisher
from rclpy.time import Time
from std_msgs.msg import String

from era_5g_interface.ros2_numpy_image import *
from era_5g_interface.task_handler_internal_q import TaskHandlerInternalQ, QueueFullAction
from fcw_core_utils.collision import *
from fcw_service.collision_worker import CollisionWorker

INPUT_TOPIC = str(os.getenv("INPUT_TOPIC", "/image_raw"))
OUTPUT_TOPIC = str(os.getenv("OUTPUT_TOPIC", "/res"))

NETAPP_INPUT_QUEUE = int(os.getenv("NETAPP_INPUT_QUEUE", 1))


class Worker(CollisionWorker):
    def __init__(
        self,
        image_queue: Queue,
        publisher: Publisher,
        config: dict,
        camera_config: dict,
        fps: float,
        viz: bool,
        viz_zmq_port: int,
        **kw
    ):
        super().__init__(
            image_queue=image_queue,
            sio=None,
            config=config,
            camera_config=camera_config,
            fps=fps,
            viz=viz,
            viz_zmq_port=viz_zmq_port,
            **kw
        )
        self.publisher = publisher

    def publish_results(self, results, metadata):
        msg = String()
        msg.data = json.dumps(results)
        self.publisher.publish(msg)


def parameters_to_dict(parameters: Dict):
    parameters_dict = {}
    for name, value in parameters.items():
        keys = name.split(".")
        dict_inner = parameters_dict
        for key in keys:
            if key not in dict_inner:
                dict_inner[key] = {}
            if key == keys[-1]:
                dict_inner[key] = value.value
            dict_inner = dict_inner[key]
    return parameters_dict


class FCWServiceNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('fcw_service_node', automatically_declare_parameters_from_overrides=True)

        self.add_on_set_parameters_callback(self.parameter_callback)
        self.config_dict = parameters_to_dict(self.get_parameters_by_prefix("config"))
        self.camera_config_dict = parameters_to_dict(self.get_parameters_by_prefix("camera_config"))
        print(self.config_dict)
        print(self.camera_config_dict)

        self.publisher = self.create_publisher(String, OUTPUT_TOPIC, 10)
        self.subscriber = self.create_subscription(Image, INPUT_TOPIC, self.image_callback, 10)

        # queue with received images
        self.image_queue = Queue(NETAPP_INPUT_QUEUE)
        self.task_handler = None
        self.worker = None
        if self.camera_config_dict:
            self.start()

    def start(self):
        if self.task_handler:
            del self.task_handler
        if self.worker:
            del self.worker

        self.image_queue.queue.clear()
        self.task_handler = TaskHandlerInternalQ(
            "fcw_service_task_handler", self.image_queue, if_queue_full=QueueFullAction.DISCARD_OLDEST
        )

        # Create worker
        self.worker = Worker(
            self.image_queue,
            self.publisher,
            self.config_dict,
            self.camera_config_dict,
            fps=self.config_dict.get("fps", 30),
            viz=self.config_dict.get("visualization", False),
            viz_zmq_port=self.config_dict.get("viz_zmq_port", 5558),
            daemon=True
        )
        self.worker.start()

    def parameter_callback(self, parameters: List[rclpy.Parameter]):
        try:
            print(parameters)
            parameters_dict = {}
            for parameter in parameters:
                parameters_dict[parameter.name] = parameter
                print(f"{parameter.name} {parameter.value}")
            parameters_dict = parameters_to_dict(parameters_dict)

            self.config_dict = parameters_dict.get("config", self.config_dict)
            self.camera_config_dict = parameters_dict.get("camera_config", self.camera_config_dict)
            print(self.config_dict)
            print(self.camera_config_dict)

            if self.camera_config_dict:
                self.start()
        except Exception as e:
            self.get_logger().error(f"Parameter callback exception: {repr(e)}")
            return SetParametersResult(successful=False)

        return SetParametersResult(successful=True)

    def image_callback(self, msg: Image):
        try:
            # Convert the ROS image message to numpy format
            np_image = image_to_numpy(msg)
        except TypeError as e:
            self.get_logger().error(f"Can't convert image to numpy. {e}")
            return
        if np_image is not None:
            metadata = {"timestamp": Time.from_msg(msg.header.stamp).nanoseconds,
                        "recv_timestamp": self.get_clock().now().nanoseconds}
            if self.task_handler is not None:
                self.task_handler.store_data(metadata, np_image)
            else:
                self.get_logger().warning("Uninitialized task handler and worker!")
        else:
            self.get_logger().warning("Empty image received!")


def main(args=None) -> None:
    """Main function."""
    rclpy.init(args=args)
    node = FCWServiceNode()

    try:
        # Spin until interrupted
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    except BaseException:
        print('Exception in node:', file=sys.stderr)
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()
        pass


if __name__ == "__main__":
    if None in [INPUT_TOPIC, OUTPUT_TOPIC]:
        print("INPUT_TOPIC and OUTPUT_TOPIC environment variables needs to be specified!")
    else:
        main()
