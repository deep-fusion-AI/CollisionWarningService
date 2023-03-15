import rclpy
from rclpy.node import Publisher
import sys, os
import cv2
from cv_bridge import CvBridge
from pathlib import Path
from typing import Union

from std_msgs.msg import String
from sensor_msgs.msg import Image

from fcw.client.client_common import CollisionWarningClient

collision_warning_client: Union[CollisionWarningClient, None] = None
publisher: Union[Publisher, None] = None

bridge = CvBridge()


# Configuration of the algorithm
config = Path("../../config/config.yaml")
# Camera settings - specific for the particular input
camera_config = Path("../../videos/video3.yaml")


def results_callback(data: dict):
    global publisher
    msg = String()
    print(data)
    msg.data = str(data)
    publisher.publish(msg)


def send_image_callback(image: Image):
    # convert received image to opencv
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    # color correction
    image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    global collision_warning_client
    # or use the HTTP transport to the /image endpoint
    collision_warning_client.send_image(image_bgr, image.header.stamp.nanosec)


def main(args=None) -> None:
    """Main function."""

    rclpy.init(args=args)

    node = rclpy.create_node('client_ros2')
    global collision_warning_client, publisher

    publisher = node.create_publisher(String, "/results", 10)
    collision_warning_client = CollisionWarningClient(config=config, camera_config=camera_config, fps=30, results_callback=results_callback)
    subscriber = node.create_subscription(Image, "/image", send_image_callback, 10)

    try:
        rclpy.spin(node)
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
    main()
