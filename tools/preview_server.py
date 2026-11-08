#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立预览中转：订阅 /image_annotated，用 stdlib HTTP 提供 MJPEG。

用法:
    source /opt/tros/humble/setup.bash
    python3 preview_server.py [--port 8001] [--topic /image_annotated]
浏览器打开 http://<板子IP>:8001/
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

BOUNDARY = "gcjpegframe"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>GC 识别预览</title>
<style>html,body{margin:0;background:#111;color:#ddd;font:14px sans-serif}
img{width:100%;max-width:640px;display:block;margin:0 auto;image-rendering:auto}
#h{padding:6px 10px;font-family:monospace}</style></head>
<body><div id="h">GC 识别预览 · /image_annotated · 原生 640x480（1:1 不放大）</div>
<img src="/stream" alt="stream">
</body></html>"""


class Latest:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = None
        self.count = 0
        self.stamp = 0.0

    def set(self, payload):
        with self.lock:
            self.data = payload
            self.count += 1
            self.stamp = time.time()

    def get(self):
        with self.lock:
            return self.data

    def health(self):
        with self.lock:
            return {"frames": self.count, "age_s": (time.time() - self.stamp) if self.stamp else -1.0}


class Relay(Node):
    def __init__(self, latest, topic):
        super().__init__("gc_preview_relay")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(CompressedImage, topic, self._cb, qos)
        self.latest = latest
        self.get_logger().info("订阅 %s，HTTP 预览已就绪" % topic)

    def _cb(self, msg):
        self.latest.set(bytes(msg.data))


def make_handler(latest):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/healthz":
                body = json.dumps(latest.health()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/last.jpg":
                data = latest.get()
                if not data:
                    self.send_error(503, "no frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = None
                try:
                    while True:
                        data = latest.get()
                        if data is not None and data is not last:
                            last = data
                            self.wfile.write(("--%s\r\nContent-Type: image/jpeg\r\n"
                                              "Content-Length: %d\r\n\r\n" % (BOUNDARY, len(data))).encode())
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                        else:
                            time.sleep(0.01)
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self.send_error(404)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--topic", default="/image_annotated")
    args = ap.parse_args()

    rclpy.init()
    latest = Latest()
    node = Relay(latest, args.topic)

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(latest))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    node.get_logger().info("HTTP 预览 http://0.0.0.0:%d/" % args.port)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()