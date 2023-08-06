import rclpy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import json
import os

# find the path where the python script is located
# and use it to find the assets folder
PATH = os.path.dirname(os.path.realpath(__file__)) + "/assets"


def publish_marker():
    rclpy.init()
    node = rclpy.create_node('marker_publisher')

    publisher = node.create_publisher(Marker, 'marker_topic', 10)

    marker = Marker()
    scale = 0.001
    marker.header.frame_id = 'world'
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.type = Marker.MESH_RESOURCE
    marker.action = Marker.ADD

    with open(f"{PATH}/pos.json") as pos:
        data = json.load(pos)
        marker.pose.position = Point(x=data["x"], y=data["y"], z=data["z"])

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale

    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 1.0

    marker.mesh_resource = f"file://{PATH}/mesh.stl"

    # with open(marker.mesh_resource.replace("file://", "")) as mesh:
    #     print(mesh.read())

    publisher.publish(marker)
    rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import time

    while True:
        publish_marker()
        time.sleep(.1)
