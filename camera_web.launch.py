from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    usb = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("hobot_usb_cam"),
                "launch",
                "hobot_usb_cam.launch.py",
            )
        ),
        launch_arguments={
            "usb_video_device": LaunchConfiguration("device"),
            "usb_image_width": LaunchConfiguration("width"),
            "usb_image_height": LaunchConfiguration("height"),
        }.items(),
    )

    web = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("websocket"),
                "launch",
                "websocket.launch.py",
            )
        ),
        launch_arguments={
            "websocket_image_topic": "/image_annotated",
            "websocket_image_type": "mjpeg",
            "websocket_only_show_image": "True",
            "websocket_output_fps": "30",
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("device", default_value="/dev/cam0"),
        DeclareLaunchArgument("width", default_value="640"),
        DeclareLaunchArgument("height", default_value="480"),
        usb,
        web,
    ])
