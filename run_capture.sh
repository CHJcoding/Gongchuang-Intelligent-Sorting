#!/bin/bash
source /opt/tros/humble/setup.bash
exec python3 /root/vision_ws/tools/capture_server.py --port 8003 "$@"
