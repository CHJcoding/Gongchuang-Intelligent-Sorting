# -*- coding: utf-8 -*-
"""只开相机 + 网页预览的 launch（不带 dnn_node_example）。

为什么要这个文件：
  v8 模型的 reg 输出是 float32，官方 dnn_node_example 的 C++ parser 按 int32 读，
  必 SIGSEGV。而我们的业务节点是**自己推理**的（USE_LOCAL_DETECTION=True），
  根本不需要 dnn_node_example。留它在链路里只会：
    1) 每 10 秒崩一次 -> 日志不再增长 -> 守护误判"卡死" -> 反复重启 -> 相机被反复复位
    2) websocket 一直报 "Websocket did not receive AI data"

所以这里只启动：
  hobot_usb_cam  -> /image (CompressedImage/mjpeg)     业务节点订阅它
  websocket      -> 网页预览 :8000，订阅 /image_annotated（我们自己画好框的图）

用法：
  GC_CAM_DEV=/dev/cam0 ros2 launch /root/vision_ws/cam_only.launch.py
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from ament_index_python import get_package_share_directory


def generate_launch_description():
    device_arg = DeclareLaunchArgument(
        "device", default_value=TextSubstitution(text="/dev/cam0"),
        description="usb camera device node")
    width_arg = DeclareLaunchArgument(
        "image_width", default_value=TextSubstitution(text="640"))
    height_arg = DeclareLaunchArgument(
        "image_height", default_value=TextSubstitution(text="480"))
    fps_arg = DeclareLaunchArgument(
        "framerate", default_value=TextSubstitution(text="30"))
    # websocket 订阅哪个话题：默认订阅业务节点画好框的图
    web_topic_arg = DeclareLaunchArgument(
        "web_image_topic", default_value=TextSubstitution(text="/image_annotated"))

    cam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("hobot_usb_cam"),
                         "launch/hobot_usb_cam.launch.py")),
        launch_arguments={
            "usb_image_width": LaunchConfiguration("image_width"),
            "usb_image_height": LaunchConfiguration("image_height"),
            "usb_framerate": LaunchConfiguration("framerate"),
            "usb_pixel_format": "mjpeg",
            "usb_video_device": LaunchConfiguration("device"),
        }.items()
    )

    web_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("websocket"),
                         "launch/websocket.launch.py")),
        launch_arguments={
            "websocket_image_topic": LaunchConfiguration("web_image_topic"),
            "websocket_image_type": "mjpeg",
            # 没有 AI 话题也不影响看图，只会在日志里提示
            "websocket_smart_topic": "/hobot_dnn_detection",
        }.items()
    )

    return LaunchDescription([
        device_arg, width_arg, height_arg, fps_arg, web_topic_arg,
        cam_node,
        web_node,
    ])